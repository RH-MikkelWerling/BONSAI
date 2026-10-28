# Complete the local tuning figure set

Run after syncing the source, on a server GPU allocation. This is additional
inference and CPU analysis, not training. National-pretraining work is untouched.
The completed main paired analysis is reused read-only in a fresh sibling
workspace, so another main bootstrap run is not required.

```bash
mkdir -p logs
completion_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
completion_root="data/paper_256_v3/analysis/local_figures_${completion_stamp}"
nohup .venv-v100/bin/python -m opera.run.complete_local_figures \
  --analysis-root data/paper_256_v3/analysis/plot_rebuild_20261028_080223 \
  --output-dir "$completion_root" \
  > "logs/local_figures_${completion_stamp}.log" 2>&1 &
tail -f "logs/local_figures_${completion_stamp}.log"
```

Use the actual completed source workspace if its timestamp differs. Do not run
this on a login node without a GPU allocation. It uses the source overnight.env,
checks the primary artifact against the selected checkpoint/config and native
prediction hashes, and prevents another completion launcher sharing the paper
root. It does not overwrite the primary predictions or cached borrowing outputs.

Steps:

1. Enrich a copy of patient metadata with real computed IPI/NCCN-IPI columns from
   the configured OPERA_PAPER_CLINICAL_METADATA or repeatable --scores tables.
   Missing group labels can be derived from observed numeric scores, with the
   two intermediate categories combined for display. Numeric scores are retained;
   missing numeric scores/components are not imputed. Conflicts stop preparation.
2. Prepare recorded-index and numeric-score comparisons against the primary OPERA
   model, on patients with that score available. Fixed availability-based priority
   selects the main clinical illustration; all available comparisons are retained.
   This fixes missing root PDFs when the old hardcoded index name IPI was absent.
3. Make five target-specific shared/post-router vector exports (one GPU evaluation
   per selected outcome, full model retained). The ordinary --outcome flag is a
   specialist-only interface and is deliberately not used. Arrows use the selected
   checkpoint's fixed shared embedding and its same projection; projection does
   not change when a target export covers only an eligible subset. All common
   calibration-map patients must have finite target vectors.
   router_projection_diagnostics.json records supported/coherent bins and shift
   magnitudes; empty arrow panels state when no coherent local shift is available.
4. Make one fresh borrowing-design GPU evaluation, exporting mean replacement
   and zero-ablation variants. Verify variant files and write a completion marker.
5. Recompute both mean-ablation primary and zero-ablation sensitivity borrowing
   summaries, including patient-bootstrap intervals, in fresh CPU cache trees.
6. Rebuild embedding/plot tables, clinical plates, figures, manuscript numbers and
   tables; write readiness, aggregate figure audit and completion.json reports.
   Available attribution/input-restriction aggregates are carried forward, explicitly
   labelled as reused, and attribution rendering is attempted when summaries exist.

## If computed IPI/NCCN-IPI are missing

The first log lines and completion.json list missing score columns/counts. Supply
real one-row-per-patient score exports with subject_id and these columns:

- ipi_computed_score (0–5), optionally ipi_computed_group;
- nccn_ipi_score (0–8), optionally nccn_ipi_group.

Add `--scores /actual/server/path/scores.parquet` to the launch command; repeat
for separate files. Alternatively supply `--score-registry /actual/registry.parquet
--score-column-map /approved/map.json` to compute them from real components with
compute_prognostic_scores. An explicit source-column map is required; the workflow
does not guess which registry fields measure the clinical components. See that
module's documentation for the schema. These source fields cannot be reconstructed
from a recorded R-IPI category alone.

If the inputs are missing, other independent stages still run and the final
status is finished_with_gaps, not complete. A national/external figure can also
remain missing until its required fits and predictions exist. This workflow
cannot promise that national pretraining is the only remaining dependency before
these reports are inspected.

## Resume the same workspace

Use the same source/config/score-file arguments, including any --scores options:

```bash
.venv-v100/bin/python -m opera.run.complete_local_figures \
  --analysis-root data/paper_256_v3/analysis/plot_rebuild_20261028_080223 \
  --output-dir "$completion_root" --resume
```

Completed stage output hashes are checked. Source code, source data or selected
checkpoint changes require a fresh workspace. Failed partial clinical render
folders are archived before retry. Fresh inference outputs belong solely to this
workspace; unfinished exports may be replaced on a retry. Detailed stage logs are
under logs/ in the completion root. Patient vector files remain server-local.

Validation is local CPU tests of selection provenance, clinical enrichment,
partial vector coverage, config generation, command orchestration and resume.
Actual GPU exports and private clinical source availability require this server run.
