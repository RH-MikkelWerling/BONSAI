# Resume the interrupted full rehearsal

From the repository root on the server, inside a GPU allocation. Sync the
updated `opera/run/complete_local_figures.py` and keep the existing
`scripts/resume_full_tuning_rehearsal.py` wrapper. Source-code freshness checks
are now disabled by default on resume, matching the full-run launcher. No
manifest update or bypass flag is needed after syncing code.

The last logged stage alone does not prove that a run stopped. Check processes
and the stage log first:

```bash
opera_full_root=data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123
ps -u "$USER" -o pid,ppid,etime,pcpu,stat,args | rg '[o]pera.run.(rerun_full_tuning_rehearsal|complete_local_figures|source_removal)'
tail -n 60 "$opera_full_root/analysis/logs/borrowing_analysis_zero.log"
```

If it has exited, resume the same workspace:

```bash
opera_resume_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
opera_resume_log="logs/full_tuning_resume_$opera_resume_stamp.log"
mkdir -p logs

nohup .venv-v100/bin/python -u scripts/resume_full_tuning_rehearsal.py \
  --output-dir "$opera_full_root" \
  > "$opera_resume_log" 2>&1 &

echo "Resume log: $opera_resume_log"
tail -f "$opera_resume_log"
```

The wrapper checks that the frozen inventories match. The nested completion
runner still checks scientific inputs and hashes of completed outputs. Code
fingerprint changes are recorded in `completion.json` under `resume_attempts`
without blocking execution or overwriting the original input fingerprint.
Optional `complete_local_figures --resume --verify-sources` restores the strict
code check. Resuming accepts completed stages from the earlier runtime. It
does not automatically recompute them when synced code changes. Complete
stages are skipped. A stage interrupted during zero-ablation analysis restarts
that analysis, retaining the completed mean analysis and inference exports.
Later IG resumes checked patient exports if its workspace already exists.
Fresh-only partial report directories are archived rather than deleted.
Downstream tables and review packages are refreshed if an upstream stage is
restarted, preventing a new summary from leaving an old packaged figure behind.

A process check and recovery lock prevent overlapping recovery attempts. The
wrapper also rejects an active original module process for the same workspace.
Logs of recovery attempts are appended separately under `logs/*.recovery.log`.
The original attempt records remain in `full_rehearsal.json`.

Scheduler walltime still applies. `nohup` does not extend an allocation.
The final status can be `finished_with_gaps` when scientific/reporting outputs
remain incomplete. Return `full_rehearsal.json`, the nested `analysis/completion.json`,
and any failed stage's log for review. This command does not open the test set.
