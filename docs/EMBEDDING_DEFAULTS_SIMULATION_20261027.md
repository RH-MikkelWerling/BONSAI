# Embedding plotting defaults: synthetic calibration

Reference counts are documented in `supervisor_validation_cohort_table1_20261007.md`:
1,980 tuning and 6,446 held-out supervised patients. The rehearsal's hash split
shows about half of tuning (~990). These are historical release reference counts,
not a fresh server inventory; individual outcomes and matched-stage intersections
may have fewer patients.

Simulations use all three sizes, a ring with an empty center, disconnected
unequal-sized clusters, two random seeds, smooth/sharp risk fields and a rare
outcome field (~0.3–3.8% risk). A quarter of statuses are independently hidden,
with inverse-observation weights. This tests kernel display behavior, not the
complete censoring process or the clinical validity of IPCW estimates. Arrows
have a known smooth vector field plus individual noise.

Sweep: bandwidth factors 0.6/1.0/1.4 (first pass), then 0.6/1.0/1.4/1.8 in
sharp/rare stress passes; effective support 10/15/25; arrow grids 5/7/9/11 with
minimum bin count 10/20/30. Metrics include surface RMSE against the known field,
patient-location coverage, gap coloring, and vector recovery. The score
RMSE + 0.15*(1-coverage) is an explicit display heuristic, not a scientific
optimization criterion. Selected defaults balance errors and geometry rather
than blindly minimizing that score.

Selected paper defaults:

- Gaussian bandwidth: 1.4 times median 50-neighbor distance, adapting to density.
- Minimum effective support: 25; retain the union-of-discs footprint and 2% peak
  density mask so broad kernels do not fill the entire plotting square.
- Arrow bins: floor(sqrt(N/30)), bounded to 5–11; about 5/8/11 for the reference
  sizes. At least 20 patients per bin and mean-vector coherence at least 0.3.
- Arrow lengths use a single labelled magnification for the whole row. Automatic
  scale targets 9% of map span at the median individual displacement, bounded
  to 1–20x; explicit router_arrow_scale=1 restores projection units. There is
  no independent length normalization per outcome.
- Existing disease contours and smooth metadata coloring remain enabled.

In the first pass, factor 1.4/support 25 gave mean visible-patient coverage
97.7% and risk RMSE 0.0448, compared with 82.2% and 0.0580 for factor 0.6 with
the same support. Factor 1.8 reduced stress-test RMSE further but colored more
of the predefined gap region; 1.4 was retained as the geometric compromise.
At N=990, a 9x9 arrow grid/minimum 20 represented 76.7% of patients versus
94.7% for 5x5. These are simulated averages, not estimates for actual maps.

Artifacts live under `scratch/embedding_defaults/`: sweep CSVs, the reproducible
`sweep.py` (run with `--scenario sharp` or `rare`), and clearly synthetic preview
images. `production_layout.png` uses the actual four-row calibration-map renderer
with synthetic inputs. None are wired into a paper-analysis recipe.

Remaining limits: bandwidth selection was not optimized on real model outcomes;
projection distortions, nonrandom censoring, highly variable IPCW weights and
very small outcome populations need inspection on the server. Effective sample
support does not establish adequate local event support for rare outcomes.
