# OPERA manuscript

`opera_ascertainment_supplement.tex` is a standalone, implementation-grounded
methods draft for first-line reconstruction, disease/arm assignment, laboratory
eligibility, administrative/infection endpoints, second-line treatment and
composites. Unresolved production branches, CTCAE versions and follow-up cutoffs
are marked explicitly. See `docs/ascertainment_review.md`; do not interpret the
draft as a verified all-v6 endpoint dictionary.

The current Table 1 is a historical layout draft from the supplied
`table1_values_long.csv` and `table1_pooled_display.csv` exports (37,467
analysis patients; 29,108 training, 1,967 tuning, 6,392 held-out). It has
**not** been refreshed against the current 37,557-patient first-line source
and must not be cited as a final cohort description. Source exports used
primary-only masking and are NOT disclosure-cleared. `tables/table1.tex` uses
the historical disease groups,
not the incomplete `Disease` field. Fine cohorts are indented under their
canonical grouped parent in the main table; one-to-one groups appear once.
Disease percentages use each column's full population, not the parent total.
High/very-high risk categories are combined
only descriptively; very-high occurred only in CLL-IPI (114 overall).

`opera_table1_supplement.tex` is a separate compilable supplement with all 24
fine cohorts, original scoring-system/risk-category counts and `arm_primary`
treatment distributions. `arm_day1_primary` is not reported. Source masks are
preserved; no missing masked counts are reconstructed. Treatment categories
labelled unresolved/unmapped/review still require clinical curation.

The supplied history span equals age because timed DOB records were included
in the history summary. The summarizer now excludes DOB/DOD, SEX and BACKGROUND
codes from clinical-history aggregation. History span and clinical-coded record
counts remain re-export placeholders; other MEDS summaries are provisional
until that corrected export is checked. Unavailable encounter/comorbidity,
diagnosis-to-treatment and prior-malignancy summaries are omitted rather than
invented. Death-event counts were not present in the supplied tables.

`opera_manuscript.tex` is the current paper entry point. The older
`opera_analysis_plan.tex` is retained as historical planning material and is
not the source of truth for the current cohort/outcome inventory.

The manuscript's primary neural argument is now a five-arm ladder on the same
pooled 69-outcome panel: plain shared multi-horizon BCE; disease-source-only
and outcome-source-only routing; both sources with additive routing; and both
sources with interactive routing. Source-only refers to adapter eligibility,
not a disease-only or outcome-only training set. The no-source router is a
matched architectural control. Residual-scale sweeps are separate from the
additive-versus-interactive comparison. The PHAIR encoder-transfer experiment
is a complementary, not yet completed, analysis. See
`docs/manuscript_result_lock_20261007.md` for the aggregate outputs needed to
turn the completed router runs into reported findings.

Compile from this directory so relative figure and bibliography paths resolve:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error opera_manuscript.tex
```

The manuscript, cohort flow, and standalone supplements use TeX Gyre Pagella
text with `mathpazo` mathematics under pdfLaTeX, matching the thesis template's
font setup. The cohort diagram labels 78,668 diagnosis-linked patients as the
pretraining population per the study team's clarification, and 37,531 as the
final supervised population. Reconcile the pretraining distinct-patient and
pre-2022 sequence counts with the frozen pretraining manifest before final
release. See `docs/opera_paper_next_actions_20261007.md` for the current run
order; `docs/manuscript_release_todo_20261004.md` is historical.

The three conceptual figures are native TikZ inputs:

- `figures/modeling_paradigms.tex`: clinical score vs tabular ML vs sequence FM;
- `figures/information_axes.tex`: patients/variables/time plus cohort/task breadth;
- `figures/readout_architectures.tex`: six multi-outcome readout designs;
- `figures/source_router.tex`: the source-anchored disease--outcome router.

The exploratory `figures/source_router_intuition.tex` is parked outside the
manuscript pending a clearer editable redesign; the original router schematic
is retained.

They share colours and drawing primitives from `opera_style.tex`, aligned with
`opera/visualization/style.py`. Keep empirical plots in Python and conceptual
schematics in TikZ. Replace `\placeholder{...}` fields only with locked results
or approved study text; placeholders deliberately prevent exploratory numbers
from becoming manuscript claims.

Use `python -m scripts.preview_publication_visualizations --output-dir
paper/render/synthetic_review` to check Python figure layouts at 1, 3, 4, 12,
13, 18, and 35 outcomes and 4, 10, and 24 diseases. Every preview is explicitly
synthetic and must never be used as a results figure. Empirical artwork should
be rerendered after the held-out analysis is locked.

The shared empirical-plot typography now prefers Arial (Liberation Sans on
Linux; DejaVu only as a fallback). The validation-only XGBoost predictability
prototype can be rerendered with
`python -m opera.run.plot_predictability_prototype --aggregates PATH_TO_CSV
--output paper/render/xgboost_predictability_validation_prototype.png`.
It is a descriptive binary-AUROC preview, not the final neural-model atlas;
the supplied aggregate CSV does not itself encode the 365-day horizon.
The plot's 17 display-only clinical groups are declared in
`opera/configs/outcome_display_groups_v1.yaml` (the historical 87 outcomes exactly once,
at most eight per group). They are intentionally distinct from the 11 broad
registry families and the grade-pairing groups used by the training loss.

Outcome KDE maps use a 450-by-450 grid and the `plasma` colormap by default.
Use `--outcome-kde-colormap inferno` to switch palettes, or
`--outcome-kde-grid-bins` to change resolution. The legacy
`--outcome-kde-bandwidth-bins` is measured against the original 90-cell grid:
resolution changes preserve smoothing in projection coordinates. White means
masked low support, not zero event fraction. Maps describe observed events
among patients with known horizon status; they are not calibrated predictions
or censoring-adjusted population risk estimates. Synthetic previews also cover
ROC comparisons with 2, 6, and 12 models, PR, calibration, decision curves,
and rarity trends against log-scaled training-event counts.

For patient scatter/facet QA, run
`python -m scripts.preview_patient_embedding_visualizations --output-dir paper/render/patient_embedding_review`.
This tests 4,000 and 30,000 patients and 2, 6, 13, and 24 cohorts. Continuous
OPERA maps now share plasma; binary classes and disease identities remain
categorical, and signed effects remain diverging. The patient CLI switches to
facets above six categories, writes at most 12 facets per page, and preserves
group colors and coordinates across pages. Centroid diagrams are restricted
to at most eight groups by default. Dense scatters use smaller markers, and
legacy outcome grids use one shared legend with contours opt-in. These are
descriptive projections, not evidence of uncertainty or causal separation.

`PRODUCTION_PLAN.md` is the display and artifact contract. It distinguishes
cohort-description work that can be completed while training runs continue
from figures that require locked held-out predictions.
