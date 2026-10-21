"""Shared GPU to-do list: priority order, exclusive claims and gating."""

import json
import threading
from argparse import Namespace
from datetime import timedelta

import pytest

import opera.run.paper_worker as worker
from test_paper_workflow import bundle, complete_run


def options(tmp_path, directory, **overrides):
    values = dict(
        bundle_dir=[directory, tmp_path / "missing_stage2"],
        runs_dir=tmp_path / "runs",
        decisions=tmp_path / "decisions.json",
        worker="w0",
        wait=False,
        poll_seconds=1,
        heartbeat_seconds=60,
        stale_minutes=30,
        max_attempts=2,
        max_consecutive_failures=3,
        max_runs=None,
        minimum_local_pretraining_epochs=20,
        minimum_dapt_epochs=20,
        external_hidden_size=256,
        local_hidden_size=256,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def fake_training(monkeypatch):
    trained = []

    def fake_train(row, path, run_dir, audit_record):
        trained.append(row)
        complete_run(run_dir, "val/x")

    monkeypatch.setattr(worker, "audit_encoder", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(worker, "train_run", fake_train)
    worker._RAW.clear()
    return trained


def test_workers_follow_scientific_priority_and_never_repeat_fits(tmp_path, fake_training):
    directory = bundle(tmp_path)
    args = options(tmp_path, directory, max_runs=3)
    assert worker.work(args) == 3
    first = [(row["phase"], row["regime"]) for row in fake_training]
    assert first[0] == ("supervision", "J")
    args.worker = "w1"
    worker.work(args)
    names = [row["name"] for row in fake_training]
    assert len(names) == len(set(names))  # no fit is trained twice
    phases = [worker.priority(row) for row in fake_training]
    assert phases == sorted(phases)
    # Without decisions only rank/query screens remain, reported as blocked.
    report = worker.status(args)
    assert report["rank_screen"] == {"blocked_needs_decision": len(
        [r for r, _, g in worker.todo(args.bundle_dir, {}) if r["phase"] == "rank_screen"]
    )}
    assert set(report["supervision/J"]) == {"complete"}


def test_decision_file_unblocks_conditional_screens_without_restart(tmp_path, fake_training):
    directory = bundle(tmp_path)
    args = options(tmp_path, directory)
    worker.work(args)
    trained = len(fake_training)
    args.decisions.write_text(json.dumps({"source_mode": "disease_only"}))
    worker.work(args)
    rank_runs = [row for row in fake_training[trained:] if row["phase"] == "rank_screen"]
    assert rank_runs
    assert all(row["condition"]["source_mode"] == "disease_only" for row in rank_runs)


def test_claims_are_exclusive_and_stale_claims_do_not_steal_live_allocations(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = []
    barrier = threading.Barrier(8)

    def attempt(index):
        claim = worker.Claim(run_dir, f"w{index}", heartbeat_seconds=3600)
        barrier.wait()
        results.append(claim.acquire())

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
    assert worker.run_state(run_dir, stale_minutes=30, max_attempts=2) == "running"
    claim = json.loads((run_dir / worker.CLAIM).read_text())
    claim["heartbeat_at"] = (worker.now() - timedelta(hours=2)).isoformat()
    (run_dir / worker.CLAIM).write_text(json.dumps(claim))
    assert worker.run_state(run_dir, stale_minutes=30, max_attempts=2) == "stale_claim"
    takeover = worker.Claim(run_dir, "new", heartbeat_seconds=3600)
    assert not takeover.acquire(steal_stale=True)
    takeover.release()
    assert json.loads((run_dir / worker.CLAIM).read_text())["worker"] != "new"


def test_unfinished_encoder_blocks_without_recording_failures(tmp_path, monkeypatch):
    directory = bundle(tmp_path)
    worker._RAW.clear()

    def not_ready(*args, **kwargs):
        raise ValueError("Wait for completed pretraining/DAPT and its metadata sidecar")

    monkeypatch.setattr(worker, "audit_encoder", not_ready)
    monkeypatch.setattr(worker, "train_run", lambda *a: pytest.fail("must not train"))
    args = options(tmp_path, directory)
    assert worker.work(args) == 0
    assert not list((tmp_path / "runs").rglob(worker.FAILED))
    assert not list((tmp_path / "runs").rglob(worker.CLAIM))


def test_failed_fits_are_retried_then_left_for_inspection(tmp_path, monkeypatch):
    directory = bundle(tmp_path)
    worker._RAW.clear()
    monkeypatch.setattr(worker, "audit_encoder", lambda *a, **k: {"status": "ok"})

    def broken(row, path, run_dir, audit_record):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(worker, "train_run", broken)
    args = options(tmp_path, directory, max_consecutive_failures=4)
    with pytest.raises(SystemExit, match="consecutive failures"):
        worker.work(args)
    failures = list((tmp_path / "runs").rglob(worker.FAILED))
    assert failures and not list((tmp_path / "runs").rglob(worker.CLAIM))
    record = json.loads(failures[0].read_text())
    assert "out of memory" in record["error"] and record["attempts"] >= 1
    run_dir = failures[0].parent
    record["attempts"] = 2
    failures[0].write_text(json.dumps(record))
    assert worker.run_state(run_dir, stale_minutes=30, max_attempts=2) == "failed"


def test_attempt_timing_feeds_the_compute_inventory(tmp_path, fake_training):
    from opera.run.paper_compute_inventory import inventory

    directory = bundle(tmp_path)
    worker.work(options(tmp_path, directory, max_runs=2))
    attempts = list((tmp_path / "runs").rglob(worker.ATTEMPTS))
    assert len(attempts) == 2
    record = json.loads(attempts[0].read_text().splitlines()[0])
    assert record["status"] == "completed" and record["wall_seconds"] >= 0
    per_fit, by_scope = inventory([directory], tmp_path / "runs")
    assert per_fit.n_attempts.sum() == 2 and per_fit.completed.sum() == 2
    assert {"origin", "regime", "gpu_hours"} <= set(by_scope)


def test_unreadable_claim_blocks_and_foreign_release_cannot_delete_owner(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / worker.CLAIM).write_text("{")
    assert worker.run_state(run_dir, stale_minutes=30, max_attempts=2) == "claim_unreadable"
    claim = worker.Claim(run_dir, "other", heartbeat_seconds=3600)
    assert not claim.acquire()
    claim.release()
    assert (run_dir / worker.CLAIM).exists()


def test_claims_are_exclusive_across_processes(tmp_path):
    import subprocess
    import sys
    import time
    run = tmp_path / "run"
    run.mkdir()
    go, stop = tmp_path / "go", tmp_path / "stop"
    code = '''
import sys,time
from pathlib import Path
from opera.run.paper_worker import Claim
run,go,stop,result=map(Path,sys.argv[1:])
while not go.exists(): time.sleep(.01)
claim=Claim(run,str(result),3600)
result.write_text(str(claim.acquire()))
while not stop.exists(): time.sleep(.01)
claim.release()
'''
    processes = [subprocess.Popen([sys.executable, "-c", code, str(run), str(go), str(stop), str(tmp_path / f"result{i}")]) for i in range(8)]
    try:
        go.touch()
        deadline = time.monotonic() + 20
        while len(list(tmp_path.glob("result*"))) < 8 and time.monotonic() < deadline:
            time.sleep(.02)
        results = [p.read_text() for p in tmp_path.glob("result*")]
        assert len(results) == 8
        assert results.count("True") == 1
    finally:
        stop.touch()
        for process in processes:
            process.wait(timeout=10)
    assert not (run / worker.CLAIM).exists()


def test_worker_rejects_missing_bundles_before_creating_runs(tmp_path):
    args = options(tmp_path, tmp_path / "absent_bundle")
    with pytest.raises(FileNotFoundError, match="Prepare a paper bundle"):
        worker.work(args)
    assert not args.runs_dir.exists()


def test_hold_file_parks_only_the_specialists(tmp_path):
    import json

    from opera.run.paper_worker import HOLD_SPECIALISTS, todo

    bundle = tmp_path / "paper" / "bundles" / "stage1"
    bundle.mkdir(parents=True)
    runs = [{"name": n, "phase": "supervision", "regime": r, "seed": 42,
             "scientific_hash": n, "condition": {}} for n, r in (("j", "J"), ("s", "S"))]
    (bundle / "bundle.json").write_text(json.dumps({"runs": runs}))
    assert all(gate is None for _, _, gate in todo([bundle], {}))
    (tmp_path / "paper" / HOLD_SPECIALISTS).touch()
    gates = {row["regime"]: gate for row, _, gate in todo([bundle], {})}
    assert gates == {"J": None, "S": "blocked_held:specialists"}


def test_throughput_prices_remaining_fits_at_the_phase_median(tmp_path):
    from collections import Counter
    from opera.run.paper_worker import ATTEMPTS, completed_wall_seconds, throughput

    run = tmp_path / "fit"
    run.mkdir()
    (run / ATTEMPTS).write_text(
        '{"status": "failed", "wall_seconds": 60}\n{"status": "completed", "wall_seconds": 540}\n'
    )
    assert completed_wall_seconds(run) == 600  # retries count toward the fit
    assert completed_wall_seconds(tmp_path / "missing") is None
    counts = Counter({("supervision/S", "complete"): 3, ("supervision/S", "pending"): 6,
                      ("supervision/J", "pending"): 1})
    report = throughput({"supervision/S": [300, 600, 900]}, counts)
    assert report["supervision/S"] == {"finished": 3, "left": 6,
                                       "median_gpu_minutes": 10.0, "gpu_hours_left": 1.0}
    assert "gpu_hours_left" not in report["supervision/J"]
    assert report["total_gpu_hours_left_priced"] == 1.0


def test_fits_without_tuning_patients_are_final_and_do_not_stop_workers(tmp_path, monkeypatch):
    """A disease with no tuning patients (NMZL) cannot have its own model; its
    fits are recorded as not estimable, never retried, and never count as
    failures, so a block of them cannot stop a worker."""
    directory = bundle(tmp_path)
    worker._RAW.clear()
    monkeypatch.setattr(worker, "audit_encoder", lambda *a, **k: {"status": "ok"})

    def no_tuning(row, path, run_dir, audit_record):
        (run_dir / worker.NOT_ESTIMABLE).write_text(json.dumps({"reason": "no tuning"}))
        raise worker.NotEstimable("Cohort 'indolent' (NMZL) has no patients in split 'tuning'.")

    monkeypatch.setattr(worker, "train_run", no_tuning)
    args = options(tmp_path, directory, max_consecutive_failures=1)
    assert worker.work(args) == 0
    assert not list((tmp_path / "runs").rglob(worker.FAILED))
    marked = list((tmp_path / "runs").rglob(worker.NOT_ESTIMABLE))
    assert marked and not list((tmp_path / "runs").rglob(worker.CLAIM))
    assert worker.run_state(marked[0].parent, stale_minutes=30, max_attempts=2) == "not_estimable"
    monkeypatch.setattr(worker, "train_run", lambda *a: pytest.fail("must not retry"))
    worker.work(args)


def test_train_run_records_a_missing_split_as_not_estimable(tmp_path, monkeypatch):
    import subprocess
    import opera.run.run_paper_queue as queue

    def fake_run(command, stdout=None, **kwargs):
        if stdout is None:  # preflight
            return subprocess.CompletedProcess(command, 0)
        stdout.write("Traceback ...\nopera.modules.datamodules.MultiCohortContrastiveDataModule."
                     "NoPatientsInSplit: Cohort 'indolent' (NMZL) has no patients in split 'tuning'.\n")
        stdout.flush()
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "queue_training.log").write_text("NoPatientsInSplit: an earlier attempt\n")
    with pytest.raises(queue.NotEstimable, match="NMZL"):
        queue.train_run({"name": "O_NMZL__all_outcomes"}, tmp_path / "c.yaml", run_dir, {})
    record = json.loads((run_dir / queue.NOT_ESTIMABLE).read_text())
    assert "split 'tuning'" in record["reason"]

    def other_failure(command, stdout=None, **kwargs):
        if stdout is not None:
            stdout.write("RuntimeError: CUDA out of memory\n")
        return subprocess.CompletedProcess(command, 0 if stdout is None else 1)

    (run_dir / queue.NOT_ESTIMABLE).unlink()
    monkeypatch.setattr(queue.subprocess, "run", other_failure)
    with pytest.raises(subprocess.CalledProcessError):
        queue.train_run({"name": "x"}, tmp_path / "c.yaml", run_dir, {})
    assert not (run_dir / queue.NOT_ESTIMABLE).exists()


def test_marking_a_disease_not_estimable_also_stops_older_workers(tmp_path):
    directory = bundle(tmp_path)
    worker._RAW.clear()
    rows = [row for row, d, _ in worker.todo([directory], {})
            if (worker.raw_config(d, row).get("supervision_factorial") or {}).get("fine_cohort")]
    assert rows, "the test bundle has disease-specific fits"
    fine = worker.raw_config(directory, rows[0])["supervision_factorial"]["fine_cohort"]
    args = options(tmp_path, directory, fine=fine, reason="no tuning patients")
    marked = worker.mark_not_estimable(args)
    assert marked
    run_dir = args.runs_dir / rows[0]["run_directory"]
    assert worker.run_state(run_dir, stale_minutes=30, max_attempts=2) == "not_estimable"
    (run_dir / worker.NOT_ESTIMABLE).unlink()  # what a worker without this code sees
    assert worker.run_state(run_dir, stale_minutes=30, max_attempts=2) == "failed"


def test_stale_claims_are_listed_and_cleared_only_on_request(tmp_path):
    directory = bundle(tmp_path)
    worker._RAW.clear()
    args = options(tmp_path, directory, execute=False)
    row = worker.todo([directory], {})[0][0]
    run_dir = args.runs_dir / row["run_directory"]
    run_dir.mkdir(parents=True)
    old = (worker.now() - timedelta(hours=2)).isoformat()
    (run_dir / worker.CLAIM).write_text(json.dumps({"host": "n1", "heartbeat_at": old}))
    assert [s["name"] for s in worker.clear_stale(args)] == [row["name"]]
    assert (run_dir / worker.CLAIM).exists()
    args.execute = True
    worker.clear_stale(args)
    assert not (run_dir / worker.CLAIM).exists()
    assert worker.run_state(run_dir, stale_minutes=30, max_attempts=2) == "pending"


def test_progress_reports_epochs_across_resumes_and_time_lost(tmp_path):
    directory = bundle(tmp_path)
    worker._RAW.clear()
    args = options(tmp_path, directory)
    row, bundle_dir, _ = next(item for item in worker.todo([directory], {})
                              if item[0]["regime"] != "S")
    monitor = worker.raw_config(bundle_dir, row)["training"]["eval_monitor_metric"]
    run_dir = args.runs_dir / row["run_directory"]
    for version, rows in enumerate(([(0, 0.5), (1, 0.4)], [(2, 0.45), (3, 0.46)])):
        logs = run_dir / "contrastive_multicohort_runs" / f"version_{version}"
        logs.mkdir(parents=True)
        lines = ["epoch,step," + monitor] + [f"{e},{e},{v}" for e, v in rows]
        (logs / "metrics.csv").write_text("\n".join(lines) + "\n")
    (run_dir / "last.ckpt").write_bytes(b"x")
    beat = (worker.now() + timedelta(minutes=50)).isoformat()
    (run_dir / worker.CLAIM).write_text(json.dumps({"host": "n1", "heartbeat_at": beat}))
    entry = next(e for e in worker.progress(args) if e["name"] == row["name"])
    assert entry["epochs_done"] == 4 and entry["best_epoch"] == 1
    assert entry["epochs_since_best"] == 2 and entry["lost_minutes"] in (49, 50)
