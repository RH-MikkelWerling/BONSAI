# OPERA manuscript

`opera_manuscript.tex` is the current paper entry point. The older
`opera_analysis_plan.tex` is retained as historical planning material and is
not the source of truth for the current cohort/outcome inventory.

Compile from this directory so relative figure and bibliography paths resolve:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error opera_manuscript.tex
```

The three conceptual figures are native TikZ inputs:

- `figures/modeling_paradigms.tex`: clinical score vs tabular ML vs sequence FM;
- `figures/information_axes.tex`: patients/variables/time plus cohort/task breadth;
- `figures/readout_architectures.tex`: the six multi-outcome readout designs.

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
