# Cross-family audit runtime, 30 October 2026

The old `opera.run.audit_tuning_router --recompute-metrics` prints its final
JSON only after all artifacts have been checked. The parallel launcher sends
that JSON to `cross_family_metrics.json.tmp` and sends stderr to
`logs/cross_family_metrics.log`. An empty temporary JSON and a quiet log do not
establish a hang. This stage reads fixed predictions and computes point metrics
on CPU. It neither fits models nor bootstraps, and a spare GPU does not help it.

An avoidable cost was found: every small prediction artifact was joined against
the entire saved estimates table. With thousands of tabular artifacts, this
repeatedly processed unrelated saved rows. The audit now constructs one unique
MultiIndex lookup and retrieves only the required estimates for each artifact.
Missing rows remain distinct from existing undefined estimates. Duplicate saved
keys still fail explicitly. The native metric checker remains unchanged.

The revised audit prints startup, timed artifact-start updates and periodic
completion counts to stderr with flushing. It continues to reserve stdout for
the final JSON. The change affects newly started processes only. No running
process or output file was modified. Ten focused audit and launcher tests pass.

On the same compute allocation as the audit, run:

```bash
ps -u "$USER" -o pid,ppid,stat,etime,time,pcpu,pmem,args \
  | grep '[o]pera.run.audit_tuning_router'
```

Run it again about one minute later. `TIME` is accumulated CPU time. If it
increases, the process is doing CPU work even with a quiet old log. `ETIME`
increases for waiting processes too. `%CPU` is supporting information and can
reflect an average since process start. A persistent `D` state indicates an
uninterruptible wait, commonly I/O. Low CPU by itself does not prove a deadlock.
If there is no matching process, inspect the parent launcher's log and step
receipt for completion or failure instead of assuming this stage is still live.

Do not restart the whole rehearsal based solely on the old audit's silence.
Inspect the process first. If a replacement is needed, restart only this CPU
audit with the updated code into separate output files. The empirical process
state on the server has not been inspected locally.
