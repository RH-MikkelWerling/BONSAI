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

`PRODUCTION_PLAN.md` is the display and artifact contract. It distinguishes
cohort-description work that can be completed while training runs continue
from figures that require locked held-out predictions.
