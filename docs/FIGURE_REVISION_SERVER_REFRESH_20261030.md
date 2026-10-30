# Server refresh for the reviewed figures

Run from the synced repository root. Use a CPU allocation with enough memory
for the saved prediction tables. This creates a fresh sibling workspace,
rebuilds clinical reports, plot tables and current routed map vectors from
saved encoder states, renders the revised templates and packages an aggregate
review ZIP. It does not train, repeat source-removal inference, run the encoder
or open the test set. There is no source-manifest freshness scan.

The command assumes the existing completed run is
`full_tuning_rehearsal_20261029_234123`. Set `opera_full_root` differently if you
continued it in a different directory.

```bash
opera_python="$PWD/.venv-v100/bin/python"
opera_full_root="$PWD/data/paper_256_v3/analysis/full_tuning_rehearsal_20261029_234123"
opera_revision_stamp=$(date +%Y%m%d_%H%M%S)
opera_revision_root="$PWD/data/paper_256_v3/analysis/figure_revision_${opera_revision_stamp}"
opera_revision_log="$PWD/logs/figure_revision_${opera_revision_stamp}.log"
mkdir -p logs

opera_revision_args=(
  --analysis-root "$opera_full_root/analysis"
  --output-dir "$opera_revision_root"
  --threads 4
  --refresh-record-gaps
)

# These are optional completed companions from this explicitly named run.
if [ -f "$opera_full_root/whole_cohort/manifest.json" ]; then
  opera_revision_args+=(--whole-cohort-dir "$opera_full_root/whole_cohort")
fi
if [ -f "$opera_full_root/ig/attributions/summary_365d/summary_manifest.json" ]; then
  opera_revision_args+=(--attribution-dir "$opera_full_root/ig/attributions")
fi

nohup "$opera_python" -u -m opera.run.refresh_reviewed_figures \
  "${opera_revision_args[@]}" > "$opera_revision_log" 2>&1 &
echo "Refresh PID: $!"
echo "Output: $opera_revision_root"
echo "Log: $opera_revision_log"
tail -f "$opera_revision_log"
```

`--refresh-record-gaps` summarises the completed 7/14/21/28-day prediction
exports across horizons and clinical families on CPU. This can take time, but
it does not repeat the long GPU inference. If those summaries have already been
refreshed, replace this flag with `--restriction-dir /explicit/completed/folder`.
That folder should contain `input_restriction_summary_by_horizon.csv`.

To include the mechanism audit from the other allocation, add this to the
argument array **before** launching, using its actual completed directory:

```bash
opera_revision_args+=(--mechanism-dir "/your/completed/parallel_audits/mechanism")
```

Likewise, an explicitly completed whole-cohort or IG directory from that other
allocation can be supplied instead of the matching optional path above. Do not
pass two competing directories for the same family. The mechanism check requires
the same fitted checkpoint and tuning split. It does not compare modification
times or list synced source files.

The full rehearsal normally copies completed IG outputs into `analysis/attributions`
as well, which is the default if no attribution argument is supplied.

## What to return

Download **only**:

```text
<opera_revision_root>/paper_review.zip
```

The workspace contains patient-level metadata and reconstructed **INTERNAL**
map vectors needed by the server renderer. Those are not included by the
aggregate packager and should stay on the server. As before, review the ZIP’s
manifest and aggregate contents under your export procedure before transferring.

`refresh_report.json` and the per-stage logs stay beside the ZIP. The final
status is `review_ready` only when required render manifests are complete and
all stages succeed. `finished_with_gaps` preserves successful work and identifies
missing inputs or failed score populations. It does not substitute an old
clinical image for an unavailable new one.

This is the updated **tuning** review. Inspect the fresh clinical counts,
router audit/provenance and final-data layouts before calling the empirical
figures final or locking the test-set evaluation.
