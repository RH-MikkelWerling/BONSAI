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


def todo(bundle_dirs, decisions):
    """Ordered, de-duplicated runs with their gating status."""
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
            elif missing:
                gate = "blocked_needs_decision:" + ",".join(sorted(missing))
            elif any(
                decisions.get(k) != v for k, v in row.get("condition", {}).items()
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


def status(args):
    decisions = load_decisions(args.decisions)
    counts = Counter()
    for row, _, gate in todo(args.bundle_dir, decisions):
        state = gate or run_state(
            args.runs_dir / row["run_directory"],
            stale_minutes=args.stale_minutes,
            max_attempts=args.max_attempts,
        )
        label = f"{row['phase']}{'/' + row['regime'] if row['phase'] == 'supervision' else ''}"
        counts[(label, state.split(":")[0])] += 1
    report = {}
    for (label, state), number in sorted(counts.items()):
        report.setdefault(label, {})[state] = number
    print(json.dumps(report, indent=2))
    return report


def next_runnable(args, decisions, audited, blocked_encoders):
    """Claim and return the first runnable fit, or None with a reason summary."""
    waiting = Counter()
    for row, bundle_dir, gate in todo(args.bundle_dir, decisions):
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
                    if row["origin"] in {"L", "RI"}
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
    parser.add_argument("command", choices=("next", "status"))
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
    parser.add_argument("--minimum-local-pretraining-epochs", type=int, default=20)
    parser.add_argument("--minimum-dapt-epochs", type=int, default=20)
    parser.add_argument("--external-hidden-size", type=int, default=256)
    parser.add_argument("--local-hidden-size", type=int, default=256)
    args = parser.parse_args()
    if args.stale_minutes * 60 <= 3 * args.heartbeat_seconds:
        parser.error("stale-minutes must comfortably exceed the heartbeat interval")
    if args.command == "status":
        status(args)
        return
    from dotenv import load_dotenv

    load_dotenv()
    work(args)


if __name__ == "__main__":
    main()
