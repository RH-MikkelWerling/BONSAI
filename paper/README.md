# OPERA manuscript

`opera_ascertainment_supplement.tex` is a standalone, implementation-grounded
methods draft for first-line reconstruction, disease/arm assignment, laboratory
eligibility, administrative/infection endpoints, second-line treatment and
composites. Unresolved production branches, CTCAE versions and follow-up cutoffs
are marked explicitly. See `docs/ascertainment_review.md`; do not interpret the
draft as a verified all-v6 endpoint dictionary.

Table 1 (`tables/table1.tex`) describes the 37,531-patient prediction cohort
(29,105 training, 1,980 tuning, 6,446 test) and was approved by the data owner
on 2026-10-20. Fine cohorts are indented under their
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

The manuscript now follows three questions: the separate and combined value of
supervised pooling across diseases/outcomes; comparison with tabular models;
and external pretraining with and without local domain adaptation. Routing is
an additional readout contrast under the same full-panel supervision. The
[full analysis plan](../docs/full_analysis_plan_20261014.md) is the current
question hierarchy and [FIGURE_OUTLINE.md](FIGURE_OUTLINE.md) is the display
storyboard. Earlier router ladders and result-lock notes are historical context.

`figures/supervision_grid.tex` shows the cell specialist, one-outcome column,
one-disease row and fully joint panel. `supervision_grid_outline.tex` is a
separate conceptual handout with the figure outline, not a results figure.
Compile it with the same command below, replacing the entry-point filename.

Table 1 is `tables/table1.tex`, approved by the data owner on 2026-10-20
(exact small counts; see the file header), with `tables/table1_by_disease.tex`
alongside; `table1.tex` in this folder renders it with the CONSORT diagram.
The superseded layouts are kept as `tables/*_original.tex`. Current cohort counts
are marked provisional and empirical results remain explicit placeholders.

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
release. See `docs/pretraining_finish_handoff_20261014.md` for the current run
order; `docs/manuscript_release_todo_20261004.md` is historical.

Restored original conceptual figures remain in the manuscript as native TikZ
inputs. The original predictability atlas and cohort flow are also retained;
the new supervision grid is an additional schematic. Final main/supplement
placement is provisional. Native inputs include:

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

Table 1 supervisor handoff: [review checklist](../docs/table1_supervisor_handoff_20261014.md).
The README previously mentioned `opera_table1_supplement.tex`, but that file is
absent from the current checkout; the by-disease table is
`tables/table1_by_disease.tex`.

The population section explicitly flushes introductory figures and cohort/table
floats in the working draft, so the cohort-flow diagram and Table 1 remain near
the population description. These draft page breaks can be relaxed during final
layout. The S/D/O/J schematic uses the original information-axes figure's rounded
cells, named axes, restrained colours and illustrative sharing lines.

The current [Results outline](RESULTS_OUTLINE.md) specifies the order, contrasts
and numbers to fill after analysis. [Figure placement](FIGURE_PLACEMENT.md)
explains the revised synthetic storyboard: routing and tuning-selected strategy
comparisons move into the main Results, while detailed calibration, clinical-score
and patient-subgroup figures remain referenced supplementary evidence.

The R-grounded definitions are in `opera_data_definitions_supplement.tex`, with
`opera_data_definitions_traceability.md` and
`opera_data_definitions_open_questions.md` as review companions. To include all
its pages in the assembled main/synthetic PDF, build it first:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=render/definitions_20261014 opera_data_definitions_supplement.tex
```

Then build the manuscript entry point. The source remains standalone; `pdfpages`
appends its PDF after the supplementary figures. Its own section/page numbering
is retained. Unresolved implementation/source questions are not resolved merely
by including the document. The unsent Table 1 supervisor message is in
`TABLE1_SUPERVISOR_REVIEW_DRAFT.md`; a refreshed current export is required.

The working Results now uses a single-column figure-first review layout: each
plate precedes its interpretation, with float barriers preventing it from
wandering into a later section. The entire working manuscript now uses a single-column layout. This is
an editorial layout for reviewing the argument, not a final venue template.
The exploratory readout catalogue (formerly Figure 11) is supplementary.
Required tables, uncertainty/provenance and selected threshold-based confusion
matrices are specified in [SUPPLEMENT_CONTENTS_PLAN.md](SUPPLEMENT_CONTENTS_PLAN.md).

For the first full prose edit, edit `opera_manuscript.tex`;
`opera_manuscript_prototype_figures.tex` only selects synthetic artwork. Edit
`opera_data_definitions_supplement.tex` for R-grounded definitions and
`opera_model_reporting_supplement.tex` for model/training/clinical-score reporting
tables and schematics. The CSVs in `supplement_templates/` are empty empirical
reporting schemas, not results. Build the reporting supplement into
`render/model_reporting_20261014` before rebuilding the assembled manuscript.

The working draft now follows Introduction → Methods → Results → Discussion.
Main Methods retain the scientific design and safeguards needed to interpret
the comparisons. `supplement_model_details.tex` holds the detailed router
formulation and hierarchical-analysis specification, included directly in the
manuscript; definitions and reporting tables remain separately editable sources.
Final section order can be adapted once a target journal is chosen.

The assembled PDF uses continuous manuscript page numbers. Footer-free embedded
wrappers (`opera_data_definitions_supplement_embedded.tex` and
`opera_model_reporting_supplement_embedded.tex`) avoid overlapping the standalone
documents' page numbers; build each wrapper into its corresponding render
directory before assembling the manuscript. Standalone sources retain their
own numbering. The TRIPOD reporting map is `TRIPOD_REPORTING_CROSSWALK.md`.

For the next working session, see [the 15 October plan](../docs/TOMORROW_PLAN_20261015.md) and [server audit](../docs/SERVER_AUDIT_20261014.md). These distinguish tested launch tooling from unfinished scientific exports.
