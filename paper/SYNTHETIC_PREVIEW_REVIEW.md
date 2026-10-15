# Synthetic manuscript review

Build from paper/:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=render/synthetic_manuscript_20261014 opera_manuscript_prototype_figures.tex
```

The wrapper uses the existing 18 prototype PDFs in render/prototypes, while the
normal manuscript continues to use figures/results. The title identifies the
synthetic preview, and the prototypes carry SYNTHETIC DESIGN PROOF marks.
The figures illustrate layout, not expected results.

Before final artwork:

- Match risk-group estimators to endpoint definitions: KM for survival and
  death-inclusive composites; AJ cumulative incidence for nonfatal events with
  competing death. The existing risk prototype has no competing events, so
  its incidence is 1 minus KM; it does not demonstrate competing mortality.
- Use NCCN-IPI/DLBCL as a focused main example, with other score/disease pairs
  separately in the supplement. Audit pre-index availability and missingness;
  never merge nominal categories across incompatible scores.
- Freeze model thresholds using training/tuning only. The old paper's cutoffs
  are design examples, not automatic thresholds for every disease/outcome.
- Numbers-at-risk tables now accompany the risk curves. Confirm follow-up
  support at two years, especially
  for recently indexed held-out patients.
- Refresh Table 1 from the sealed population and obtain supervisor sign-off.
- Add paired seed and compute summaries: epochs alone cannot compare the
  speed or efficiency of 64- and 256-wide models.
- Keep interval types explicit: clinical-group IQR bars describe heterogeneity,
  whereas paired bootstrap intervals describe uncertainty.
- Review final main/supplement placement: the complete synthetic assembly is
  a storyboard, not yet a journal-sized figure selection.

The updated risk/clinical-score prototypes follow the DLBCL death-inclusive
treatment-failure example and plot event-free survival. Baseline risk groups
use the historical candidate cutoffs (0.10, 0.30, 0.65). Both figures use the
same synthetic evaluation patients; the clinical operating threshold is locked
using a separate synthetic tuning sample. Final cutoffs must be justified using
the low-risk comparability and meaningful-group-size rationale on actual tuning
data. No synthetic low-risk comparison establishes that criterion clinically.
