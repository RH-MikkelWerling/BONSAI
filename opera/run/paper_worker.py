"""Shared to-do list for paper fits: every GPU job runs the same worker.

Start one worker per GPU allocation (for example one qsub job per terminal).
Each worker repeatedly claims the highest-priority fit that nobody else is
running, trains it, and moves on. Workers coordinate only through claim files
in the shared run directory, so they may run on different nodes.

Claiming uses an exclusive-create file (atomic on shared filesystems, unlike
``flock`` across nodes). A background heartbeat refreshes the claim while the
fit trains. If a job is killed, its claim is reported stale after ``--stale-minutes``.
It remains reserved until the scheduler confirms the whole allocation stopped
and its claim is manually cleared; resuming uses the run's own last.ckpt. Completed fits are recognised by their verified selection
record and never retrained. Phases that need a tuning decision (rank/query
screens) or a stage-2 bundle stay blocked until those files exist; the worker
re-reads them on every pass, so no restart is needed after a decision.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bonsai.functional.checkpointing import verified_training_selection
from opera.run.run_paper_queue import (
    NOT_ESTIMABLE,
    NotEstimable,
    audit_encoder,
    encoder_key,
    load_run_config,
    train_run,
)

# Scientific priority: shared/pooled fits and the bounded source screen first
# (they gate the router decisions), conditional screens and stage-2 controls
# as soon as they unlock, and the large specialist panel last.
PHASE_ORDER = (
    ("supervision", "J"),
    ("supervision", "D"),
    ("supervision", "O"),
    ("pretraining_control", None),
    ("source_screen", None),
    ("rank_screen", None),
    ("query_screen", None),
    ("final_sources", None),
    ("router_controls", None),
    ("external", None),
    ("data_efficiency", None),
    ("supervision", "S"),
)
DECISIONS_REQUIRED = {
    "rank_screen": {"source_mode"},
    "query_screen": {"source_mode", "rank"},
}
CLAIM = ".worker_claim.json"
FAILED = "worker_failed.json"


def now():
    return datetime.now(timezone.utc)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


_RAW = {}


def raw_config(bundle_dir, row):
    path = Path(bundle_dir) / row["config_path"]
    if path not in _RAW:
        import yaml

        _RAW[path] = yaml.safe_load(path.read_text())
    return _RAW[path]


HOLD_SPECIALISTS = "HOLD_SPECIALISTS"


def priority(row):
    for rank, (phase, regime) in enumerate(PHASE_ORDER):
        if row["phase"] == phase and (regime is None or row["regime"] == regime):
            return rank
    return len(PHASE_ORDER)


def todo(bundle_dirs, decisions, speculative=False):
    """Ordered, de-duplicated runs with their gating status.

    ``speculative``: screening branches whose deciding choice is not recorded
    yet run instead of waiting; once a choice is recorded, the branches it
    excludes drop out as usual (fits already finished stay on disk, unused).
    The decisions themselves are still taken in order, on tuning predictions.
    """
    rows, seen = [], set()
    for bundle_dir in bundle_dirs:
        bundle_dir = Path(bundle_dir)
        manifest = bundle_dir / "bundle.json"
        if not manifest.is_file():
            continue
        for row in json.loads(manifest.read_text())["runs"]:
            if row["scientific_hash"] in seen:
                continue
            missing = DECISIONS_REQUIRED.get(row["phase"], set()) - set(decisions)
            # `touch <paper root>/HOLD_SPECIALISTS` parks the specialists while
            # everything else runs; removing the file releases them (no restart).
            held = ((bundle_dir.parent.parent / HOLD_SPECIALISTS).exists()
                    and row["phase"] == "supervision" and row["regime"] == "S")
            if held:
                gate = "blocked_held:specialists"
            elif missing and not speculative:
                gate = "blocked_needs_decision:" + ",".join(sorted(missing))
            elif any(
                k in decisions and decisions[k] != v
                for k, v in row.get("condition", {}).items()
            ):
                continue  # A screening branch that the recorded decision excluded.
            else:
                gate = None
            seen.add(row["scientific_hash"])
            rows.append((row, bundle_dir, gate))
    rows.sort(key=lambda item: (priority(item[0]), item[0]["seed"], item[0]["name"]))
    return rows


def run_state(run_dir, *, stale_minutes, max_attempts):
    if (run_dir / "training_complete.json").is_file():
        try:
            verified_training_selection(run_dir)
        except ValueError as error:
            return f"complete_unverified:{error}"
        return "complete"
    if (run_dir / NOT_ESTIMABLE).is_file():
        return "not_estimable"  # Final: no data to fit or select on, by design.
    claim = read_json(run_dir / CLAIM)
    if (run_dir / CLAIM).exists() and not claim:
        return "claim_unreadable"
    if claim:
        try:
            heartbeat = datetime.fromisoformat(claim["heartbeat_at"])
            if heartbeat.tzinfo is None:
                return "claim_unreadable"
        except (KeyError, TypeError, ValueError):
            return "claim_unreadable"
        if (now() - heartbeat).total_seconds() < stale_minutes * 60:
            return "running"
        return "stale_claim"
    failure = read_json(run_dir / FAILED)
    if failure and failure.get("attempts", 0) >= max_attempts:
        return "failed"
    return "pending"


class Claim:
    """Exclusive claim on one run directory with a heartbeat thread."""

    def __init__(self, run_dir, worker, heartbeat_seconds):
        self.path = Path(run_dir) / CLAIM
        self.worker = worker
        self.token = uuid.uuid4().hex
        self.heartbeat_seconds = heartbeat_seconds
        self._stop = threading.Event()
        self._thread = None

    def _payload(self, claimed_at):
        return {
            "token": self.token,
            "worker": self.worker,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "claimed_at": claimed_at,
            "heartbeat_at": now().isoformat(),
        }

    def acquire(self, *, steal_stale=False):
        if steal_stale:
            # An expired heartbeat does not prove training stopped on another node.
            return False
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        self.claimed_at = now().isoformat()
        with os.fdopen(fd, "w") as handle:
            json.dump(self._payload(self.claimed_at), handle)
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return True

    def _beat(self):
        while not self._stop.wait(self.heartbeat_seconds):
            if (read_json(self.path) or {}).get("token") != self.token:
                return
            temporary = self.path.with_name(f"{CLAIM}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(self._payload(self.claimed_at)))
            os.replace(temporary, self.path)

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if (read_json(self.path) or {}).get("token") == self.token:
            self.path.unlink(missing_ok=True)


ATTEMPTS = "worker_attempts.jsonl"


def gpu_name():
    """Best-effort GPU model for the compute inventory; never fails a fit."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        names = result.stdout.strip().splitlines()
        index = int(visible) if visible.isdigit() and int(visible) < len(names) else 0
        return names[index] if names else None
    except Exception:
        return None


def record_attempt(run_dir, worker, started, status):
    """Append wall-clock time of one training attempt (GPU-hour inventory)."""
    finished = now()
    with (Path(run_dir) / ATTEMPTS).open("a") as handle:
        handle.write(json.dumps({
            "worker": worker,
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": gpu_name(),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "wall_seconds": (finished - started).total_seconds(),
            "status": status,
        }) + "\n")


def record_failure(run_dir, worker, error):
    path = Path(run_dir) / FAILED
    previous = read_json(path) or {}
    path.write_text(
        json.dumps(
            {
                "attempts": int(previous.get("attempts", 0)) + 1,
                "last_worker": worker,
                "last_failed_at": now().isoformat(),
                "error": repr(error),
                "traceback": traceback.format_exc(limit=8),
                "log": str(Path(run_dir) / "queue_training.log"),
            },
            indent=2,
        )
        + "\n"
    )


def load_decisions(path):
    if not path:
        return {}
    value = read_json(path)
    if value is None and Path(path).exists():
        raise ValueError(f"Decisions file is not valid JSON: {path}")
    return value or {}


def completed_wall_seconds(run_dir):
    """Wall time of a fit's completing attempt plus any earlier ones, or None."""
    path = Path(run_dir) / ATTEMPTS
    try:
        attempts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not any(a.get("status") == "completed" for a in attempts):
        return None
    return sum(float(a.get("wall_seconds", 0)) for a in attempts)


def throughput(walls, counts):
    """Median GPU-minutes per finished fit and GPU-hours left, per phase.

    Fits still to do are priced at their phase's median; a phase with no
    finished fit yet has no projection.
    """
    report = {}
    for label in sorted({label for label, _ in counts}):
        times = sorted(walls.get(label, []))
        left = sum(n for (lab, state), n in counts.items()
                   if lab == label and state not in {"complete", "not_estimable"})
        entry = {"finished": len(times), "left": left}
        if times:
            median = times[len(times) // 2]
            entry["median_gpu_minutes"] = round(median / 60, 1)
            entry["gpu_hours_left"] = round(left * median / 3600, 1)
        report[label] = entry
    known = [e["gpu_hours_left"] for e in report.values() if "gpu_hours_left" in e]
    report["total_gpu_hours_left_priced"] = round(sum(known), 1)
    return report


def status(args):
    decisions = load_decisions(args.decisions)
    counts, walls = Counter(), {}
    for row, _, gate in todo(args.bundle_dir, decisions, getattr(args, "speculative", False)):
        run_dir = args.runs_dir / row["run_directory"]
        state = gate or run_state(
            run_dir,
            stale_minutes=args.stale_minutes,
            max_attempts=args.max_attempts,
        )
        label = f"{row['phase']}{'/' + row['regime'] if row['phase'] == 'supervision' else ''}"
        counts[(label, state.split(":")[0])] += 1
        wall = completed_wall_seconds(run_dir)
        if wall is not None:
            walls.setdefault(label, []).append(wall)
    report = {}
    for (label, state), number in sorted(counts.items()):
        report.setdefault(label, {})[state] = number
    print(json.dumps(report, indent=2))
    print(json.dumps({"throughput": throughput(walls, counts)}, indent=2))
    return report


def next_runnable(args, decisions, audited, blocked_encoders):
    """Claim and return the first runnable fit, or None with a reason summary."""
    waiting = Counter()
    for row, bundle_dir, gate in todo(args.bundle_dir, decisions, getattr(args, "speculative", False)):
        if gate:
            waiting[gate] += 1
            continue
        run_dir = args.runs_dir / row["run_directory"]
        state = run_state(
            run_dir, stale_minutes=args.stale_minutes, max_attempts=args.max_attempts
        )
        if state != "pending":
            if state in {"running", "stale_claim", "claim_unreadable"}:
                waiting["running_elsewhere" if state == "running" else state] += 1
            continue
        # Bundle configs carry frozen absolute paths, so the encoder identity is
        # read cheaply; Hydra resolution happens only for audits and claims.
        raw = raw_config(bundle_dir, row)
        key = encoder_key(raw, row)
        if key in blocked_encoders:
            waiting["blocked_encoder_not_ready"] += 1
            continue
        if key not in audited:
            try:
                _, _, resolved = load_run_config(row, bundle_dir)
                audited[key] = audit_encoder(
                    resolved,
                    row["origin"],
                    expected_width=args.local_hidden_size
                    if row["origin"] in {"L", "RI", "FZ"}
                    else args.external_hidden_size,
                    minimum_local_epochs=args.minimum_local_pretraining_epochs,
                    minimum_dapt_epochs=args.minimum_dapt_epochs,
                )
            except (ValueError, FileNotFoundError) as error:
                blocked_encoders[key] = str(error)
                waiting["blocked_encoder_not_ready"] += 1
                continue
        run_dir.mkdir(parents=True, exist_ok=True)
        claim = Claim(run_dir, args.worker, args.heartbeat_seconds)
        if claim.acquire():
            if (run_dir / "training_complete.json").is_file():
                claim.release()
                continue
            try:
                path, _, _ = load_run_config(row, bundle_dir)  # verifies hashes
            except Exception:
                claim.release()
                raise
            return row, path, run_dir, audited[key], claim, waiting
        waiting["claimed_during_scan"] += 1
    return None, None, None, None, None, waiting


def epoch_progress(run_dir, monitor):
    """Validated epochs across every resume, the best epoch and the last save."""
    import pandas as pd

    frames = []
    for path in sorted(Path(run_dir).glob("contrastive_multicohort_runs/*/metrics.csv")):
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        if monitor in frame and "epoch" in frame:
            frames.append(frame[["epoch", monitor]].dropna())
    result = {"epochs_done": 0, "best_epoch": None, "epochs_since_best": None}
    if frames:
        scores = pd.concat(frames).groupby("epoch")[monitor].last()
        best = scores.idxmax() if "auc" in monitor.lower() else scores.idxmin()
        result.update(epochs_done=int(scores.index.max()) + 1, best_epoch=int(best),
                      epochs_since_best=int(scores.index.max() - best))
    last = Path(run_dir) / "last.ckpt"
    result["last_checkpoint_at"] = (
        datetime.fromtimestamp(last.stat().st_mtime, timezone.utc).isoformat()
        if last.is_file() else None)
    return result


def progress(args):
    """Where each unfinished non-specialist fit stands, and what a kill lost.

    ``lost_minutes`` is the time between the fit's last saved checkpoint and its
    last heartbeat: training a killed job did that a resume repeats. Epochs left
    is an upper bound from the epoch budget and early-stopping patience.
    """
    report, specialists = [], Counter()
    for row, bundle_dir, gate in todo(args.bundle_dir, load_decisions(args.decisions), getattr(args, "speculative", False)):
        run_dir = args.runs_dir / row["run_directory"]
        if row["phase"] == "supervision" and row["regime"] == "S":
            # Specialists are only counted, and only with --specialists: checking
            # each of ~1,650 run directories on a network file system is slow.
            if getattr(args, "specialists", False):
                state = gate or run_state(run_dir, stale_minutes=args.stale_minutes,
                                          max_attempts=args.max_attempts)
                specialists[state.split(":")[0]] += 1
            continue
        state = gate or run_state(run_dir, stale_minutes=args.stale_minutes,
                                  max_attempts=args.max_attempts)
        if state in {"complete", "not_estimable"} or state.startswith("blocked"):
            continue
        training = raw_config(bundle_dir, row)["training"]
        entry = {"name": row["name"], "phase": row["phase"], "state": state,
                 "run_directory": str(run_dir),
                 **epoch_progress(run_dir, training["eval_monitor_metric"])}
        if entry["epochs_since_best"] is not None:
            entry["max_epochs_left"] = min(
                int(training["epochs"]) - entry["epochs_done"],
                int(training.get("early_stopping_patience") or 10**6)
                - entry["epochs_since_best"])
        claim = read_json(run_dir / CLAIM) or {}
        entry["host"], entry["last_heartbeat"] = claim.get("host"), claim.get("heartbeat_at")
        if entry["last_heartbeat"] and entry["last_checkpoint_at"]:
            lost = (datetime.fromisoformat(entry["last_heartbeat"])
                    - datetime.fromisoformat(entry["last_checkpoint_at"]))
            entry["lost_minutes"] = round(max(lost.total_seconds(), 0) / 60)
        report.append(entry)
    print(json.dumps({"long_fits": report, "specialists": dict(specialists)}, indent=2))
    return report


def clear_stale(args):
    """List stale claims (killed jobs, e.g. walltime); remove them with --execute.

    Only run --execute once the jobs that held them have stopped (qstat): a
    heartbeat older than --stale-minutes means the worker stopped beating,
    which after a walltime kill is certain. The next worker resumes each fit
    from its own last.ckpt.
    """
    stale = []
    for row, _, _ in todo(args.bundle_dir, load_decisions(args.decisions), getattr(args, "speculative", False)):
        run_dir = args.runs_dir / row["run_directory"]
        state = run_state(run_dir, stale_minutes=args.stale_minutes,
                          max_attempts=args.max_attempts)
        if state not in {"stale_claim", "claim_unreadable"}:
            continue
        claim = read_json(run_dir / CLAIM) or {}
        stale.append({"name": row["name"], "state": state, "host": claim.get("host"),
                      "worker": claim.get("worker"), "heartbeat_at": claim.get("heartbeat_at"),
                      "resumes_from_last_ckpt": (run_dir / "last.ckpt").is_file()})
        if args.execute:
            # A killed job never records its attempt; log it from the claim so
            # the compute inventory counts the GPU time it used.
            try:
                started = datetime.fromisoformat(claim["claimed_at"])
                ended = datetime.fromisoformat(claim["heartbeat_at"])
            except (KeyError, TypeError, ValueError):
                started = ended = None
            if started and ended:
                with (run_dir / ATTEMPTS).open("a") as handle:
                    handle.write(json.dumps({
                        "worker": claim.get("worker"), "host": claim.get("host"),
                        "cuda_visible_devices": claim.get("gpu"), "gpu": None,
                        "started_at": started.isoformat(), "finished_at": ended.isoformat(),
                        "wall_seconds": max((ended - started).total_seconds(), 0.0),
                        "status": "killed",
                    }) + "\n")
            (run_dir / CLAIM).unlink(missing_ok=True)
    print(json.dumps({"stale_claims": stale, "cleared": bool(args.execute)}, indent=2))
    return stale


def mark_not_estimable(args):
    """Record a disease's own fits (O and S) as not estimable before they run.

    For a disease with no patients in a split the fits need (NMZL has none in
    the tuning year). Workers started before this code existed skip the fits
    because the failure record is past any attempt limit; current workers read
    ``not_estimable.json``. Fits already complete or running are left alone.
    """
    marked = []
    for row, bundle_dir, _ in todo(args.bundle_dir, {}):
        factorial = raw_config(bundle_dir, row).get("supervision_factorial") or {}
        if row["phase"] != "supervision" or factorial.get("fine_cohort") != args.fine:
            continue
        run_dir = args.runs_dir / row["run_directory"]
        state = run_state(run_dir, stale_minutes=args.stale_minutes,
                          max_attempts=args.max_attempts)
        if state in {"complete", "running"}:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {"name": row["name"], "reason": args.reason, "marked_at": now().isoformat()}
        (run_dir / NOT_ESTIMABLE).write_text(json.dumps(record, indent=2) + "\n")
        (run_dir / FAILED).write_text(json.dumps(
            {"attempts": 999, "error": f"not estimable: {args.reason}"} | record, indent=2) + "\n")
        marked.append(row["name"])
    print(json.dumps({"fine_cohort": args.fine, "marked": len(marked), "fits": marked}, indent=2))
    return marked


def work(args):
    if not any((Path(d) / "bundle.json").is_file() for d in args.bundle_dir):
        raise FileNotFoundError("Prepare a paper bundle before starting GPU workers")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    audited, completed, consecutive_failures = {}, 0, 0
    while True:
        decisions = load_decisions(args.decisions)
        blocked_encoders = {}  # Re-audited each pass: pretraining may have finished.
        row, path, run_dir, audit_record, claim, waiting = next_runnable(
            args, decisions, audited, blocked_encoders
        )
        if row is None:
            summary = {
                "worker": args.worker,
                "completed_this_session": completed,
                "waiting": dict(waiting),
            }
            if blocked_encoders:
                summary["encoder_blockers"] = sorted(set(blocked_encoders.values()))
            if not waiting or not args.wait:
                print(
                    json.dumps(summary | {"status": "no_runnable_fits"}, indent=2),
                    flush=True,
                )
                return completed
            print(json.dumps(summary | {"status": "waiting"}), flush=True)
            time.sleep(args.poll_seconds)
            continue
        print(
            f"[{args.worker}] claimed {row['phase']} {row['name']} -> {run_dir}",
            flush=True,
        )
        started, status = now(), "failed"
        try:
            train_run(row, path, run_dir, audit_record)
            status = "completed"
        except NotEstimable as reason:  # A property of the data, not a failure.
            status = "not_estimable"
            (Path(run_dir) / FAILED).unlink(missing_ok=True)
            print(f"[{args.worker}] not estimable {row['name']}: {reason}", flush=True)
        except Exception as error:  # Record, release and continue with other fits.
            record_failure(run_dir, args.worker, error)
            consecutive_failures += 1
            print(f"[{args.worker}] FAILED {row['name']}: {error!r}", flush=True)
            if consecutive_failures >= args.max_consecutive_failures:
                raise SystemExit(
                    f"{consecutive_failures} consecutive failures; stopping this worker "
                    "(check GPU/environment and the run logs)"
                ) from error
        else:
            completed += 1
            consecutive_failures = 0
            (Path(run_dir) / FAILED).unlink(missing_ok=True)
            print(f"[{args.worker}] completed {row['name']}", flush=True)
        finally:
            try:
                record_attempt(run_dir, args.worker, started, status)
            finally:
                claim.release()
        if args.max_runs and completed >= args.max_runs:
            return completed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("next", "status", "mark-not-estimable", "clear-stale", "progress"))
    parser.add_argument("--execute", action="store_true",
                        help="clear-stale: remove the stale claims (default: only list them)")
    parser.add_argument("--fine", help="mark-not-estimable: the disease (cohort_fine)")
    parser.add_argument("--reason", help="mark-not-estimable: why, recorded with each fit")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        action="append",
        required=True,
        help="Stage-1 bundle; repeat for the stage-2 bundle (may not exist yet)",
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--decisions", type=Path, help="Tuning decisions JSON, re-read every pass"
    )
    parser.add_argument(
        "--speculative", action="store_true",
        help="Run rank/query screen branches before their deciding choice is recorded; "
             "recorded choices still prune the excluded branches",
    )
    parser.add_argument("--specialists", action="store_true",
                        help="progress: also count the specialists' states (slow on network storage)")
    parser.add_argument("--worker", default=f"{socket.gethostname()}-{os.getpid()}")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Keep polling while fits are blocked or running elsewhere",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--stale-minutes", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--minimum-local-pretraining-epochs", type=int, default=10)
    parser.add_argument("--minimum-dapt-epochs", type=int, default=10)
    parser.add_argument("--external-hidden-size", type=int, default=256)
    parser.add_argument("--local-hidden-size", type=int, default=256)
    args = parser.parse_args()
    if args.stale_minutes * 60 <= 3 * args.heartbeat_seconds:
        parser.error("stale-minutes must comfortably exceed the heartbeat interval")
    if args.command == "status":
        status(args)
        return
    if args.command == "progress":
        progress(args)
        return
    if args.command == "clear-stale":
        clear_stale(args)
        return
    if args.command == "mark-not-estimable":
        if not (args.fine and args.reason):
            parser.error("mark-not-estimable needs --fine and --reason")
        mark_not_estimable(args)
        return
    from dotenv import load_dotenv

    load_dotenv()
    work(args)


if __name__ == "__main__":
    main()
