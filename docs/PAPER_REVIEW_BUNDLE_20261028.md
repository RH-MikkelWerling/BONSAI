# Bring completed paper outputs down from the server

Sync the current code first. These commands render the available aggregate inputs
and package a fresh review ZIP. They do not restart inference, attribution,
clinical prediction exports or source removal. Missing families are recorded,
so unfinished runs can continue independently.

Run from the repository root on the server:

```bash
figure_root=data/paper_256_v3/analysis/local_figures_20261028_143127
bundle_stamp=$(TZ=Europe/Copenhagen date +%Y%m%d_%H%M%S)
review_root="data/paper_256_v3/analysis/paper_review_download_${bundle_stamp}"

.venv-v100/bin/python -m opera.run.assemble_figure_review \
  --analysis-root "$figure_root" \
  --output-dir "$review_root"

.venv-v100/bin/python -m opera.run.package_figure_review \
  --review-dir "$review_root" \
  --analysis-root "$figure_root" \
  --output "${review_root}.zip"
```

Download the printed ZIP through the existing file-transfer application. Put it
in `scratch/paper_reading_20261028/` locally (or provide its local path). We can
then inspect the figures, contrast summaries, mean/zero borrowing results and
missing-input manifests together. The bundle deliberately supports a partial
review, not a declaration that all analyses have finished.

If a refreshed clinical workspace is already complete, pass its directory to
the assembly command, for example:

```bash
  --clinical-dir data/paper_256_v3/analysis/clinical_training_groups_fixed_20261028/clinical_scores
```

For the running launcher, the explicit workspaces are recorded in
`logs/finish_figure_review_<timestamp>.tsv.paths.json`. When ready, add the
actual paths as `--clinical-dir <clinical_root>/clinical_scores`,
`--attribution-dir <ig_root>/attributions` and
`--restriction-dir <ig_root>/input_restrictions` to the assembly command.
Use a new timestamp/destination for each bundle. Do not guess that an older
clinical directory uses the current training-derived groups.

The archive contains PDF/PNG figures, selected renderer/provenance manifests,
explicitly named aggregate result tables and `bundle_manifest.json` with file
hashes and missing families. It excludes unknown tables, raw predictions,
checkpoints, embeddings, patient arrays and INTERNAL files. Aggregate tables
with recognised patient-ID columns are rejected. Existing disclosure review
requirements still apply to figures and aggregate counts.

This ZIP is for manuscript review, not a backup of the trained models or a
national-checkpoint transfer. The first bundle is useful immediately; generate
a second after the remaining stages finish.
