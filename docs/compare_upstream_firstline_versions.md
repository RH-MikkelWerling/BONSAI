# Compare September treatment and MM V5 inputs before first-line rerun

Do not switch to the newest file by filename alone. Run the aggregate-only
comparison on the secure server; it reads but does not change the Parquets and
emits no patient IDs. Supply the four absolute paths, ideally to restricted
copies rather than live collaborator paths:

```bash
python -m opera.run.compare_upstream_firstline_parquets \
  --old-treatment '/absolute/copied/table_treatment_clean_2026_04_05.parquet' \
  --new-treatment '/absolute/copied/table_treatment_clean_2026_09_10.parquet' \
  --old-mm '/absolute/copied/MM_V4_5lines.parquet' \
  --new-mm '/absolute/copied/MM_V5_100_lines.parquet' \
  --output "$PWD/data/results/upstream_firstline_versions_$(date -u +%Y%m%dT%H%M%SZ)_INTERNAL.json"
```

Review treatment schemas, row/patient overlap, date range, counts by year,
source and treatment code, and missingness. A smaller September file is
plausibly a cleaner administered-treatment extract, but could also omit a
source or time period; the comparison does not adjudicate which. For MM, total
rows in a 100-line and a 5-line table are inherently incomparable. The script
compares their valid line-1 patient sets and selected dates (plus regimen when
available). A new seed can change the cohort even if its total size barely
changes.

Only after these results are reviewed should `OPERA_FIRSTLINE_MM_SEED` and
`OPERA_FIRSTLINE_TREATMENT` point at a specific frozen version. Keep both old
and new files, hashes, and the comparison report in the release record. Even
if V5/September are adopted, regenerate first line, second line, outcomes and
index-based features consistently; do not mix them with v4 cohort outputs.
