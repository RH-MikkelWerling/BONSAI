# OPERA paper figure system

The visual direction is restrained and precise: white space, dark slate type,
an indigo model identity, quiet structural lines, and explicit uncertainty.
The first thing a reader should see is the scientific comparison.

The user-selected baseline, confirmed on 30 October 2026, is the figure system
from **before the LDH conversion**. Preserve its heatmaps, interval forests,
scatter plots, survival plates and illustrated schematics. The simplified LDH
edition is not the preferred visual direction. Improve precision, spacing and
readability within the original style, and keep proposed compositions clearly
distinguishable from original figures and from new empirical results.

## Shared rules

- Import `opera.visualization.style`; use its sizes, colors and export helper.
  DejaVu Sans is bundled with Matplotlib, keeping layout consistent across hosts.
- Use left-aligned panel titles, bold panel letters, and short axis labels with
  units. Keep interpretation and methodological detail in the caption.
- Omit figure subtitles and descriptive footer prose. Include the horizon
  compactly in the title when needed to identify the prediction task. Counts
  belong beside legend categories; eligibility and support details remain in
  the diagnostic artifacts. The shared title helper accepts legacy subtitle
  arguments but does not render them.
- Preserve model identity across figures: OPERA indigo, DAPT periwinkle,
  tabular + IPI green, tabular EHR teal, IPI sienna, pretraining slate.
  Use `model_color` and `model_linestyle` for comparisons. Unknown identities
  use deterministic mappings; finite palettes can collide, so inspect legends.
- Reserve ribbons for supplied uncertainty. State the interval definition and
  resampling unit in the caption. A seed range, SD, bootstrap CI and posterior
  credible interval are different quantities.
- Use sequential color for ordered magnitudes and a centered diverging scale
  for signed effects. Gray means unavailable, never zero. Share normalization
  whenever panels or point groups share one colorbar.
- Favor points and intervals for performance comparisons. Keep empirical ROC
  and precision–recall curves unsmoothed. Do not add decorative area fills.
- Preserve all models, outcomes and negative effects. When a divergent treat-all
  reference extends below the displayed decision-curve range, label that fact;
  retain the entire observed model curve.
- Use unobtrusive grids only where they help compare values. Avoid grids and
  numeric ticks on abstract embedding maps unless coordinates serve a specific
  diagnostic purpose. Reuse coordinates and bounds for metadata comparisons.
- Keep legends outside dense data where space allows. Place displaced cluster
  labels with connectors to their original coordinates.

## Export and final size

`save_fig(fig, "figure.png")` writes a 300-dpi preview and a vector PDF.
`save_fig(fig, "figure.svg")` writes SVG with editable text. PDF embeds TrueType
fonts. Dense scatter layers are rasterized at the requested export resolution;
labels, lines, and intervals remain vector. These rules apply to all plot
families using the shared helper, including attribution plots.

The existing width constants are approximately 3.46 inches for one column and
7.09 inches for two columns. Match the eventual venue template before assembly.
Tight cropping and external legends can change exported dimensions. Inspect
the *placed* figure: scaling a 10-inch diagnostic to 7 inches also scales type.
Split dense diagnostic grids into separate manuscript panels where necessary.

## Coverage of this pass

| Family | Changes and checks |
| --- | --- |
| Classification | Shared typography; uncluttered curves; full model comparisons; aligned calibration distributions; readable decision curves |
| Survival | Stable line encodings; ordered horizons and calibration bins; matched treat-all references |
| Model comparison | Correct metric references; consistent row order; neutral numeric estimate column |
| Outcome and disease transfer | Consistent scales; missing-reference masking; readable labels; source/target labels |
| Embedding and training | Shared style/export; no eight-outcome truncation in training curves |
| Censoring-aware maps | Compact grid; common bounds and time normalization; separate legends; support annotation |
| Rarity and hierarchical rarity | Shared style/export; deterministic outcome colors and line patterns |
| Patient benefit and treatment atlas | Shared typography, palette roles, and vector export; existing specialized layouts retained |
| Vocabulary atlas | Shared style/export; displaced centroid labels with connectors |

## Review gallery

Run `python -m opera.run.render_style_gallery` and open
`outputs/style_gallery/index.html`. It contains deterministic synthetic
proofs, with PNG, PDF and SVG for each. Every artifact is marked **SYNTHETIC ·
DESIGN PROOF**. This is a layout review, not evidence of model performance.

Existing result artifacts are not rewritten by changing the style module.
Regenerate manuscript figures from the actual evaluation outputs, then review
long clinical labels, crowded cohorts, missing cells and uncertainty at final
placement size. A synthetic gallery cannot certify every real-data layout.

## Atlas and pipeline updates

Patient metadata atlases use equal map bounds within a stage, frameless axes,
full-range continuous color scales, and separate legends with every displayed
category and its count. Category draw order is shuffled deterministically to
avoid consistently covering one group with another. The patient-embedding CLI
now writes `patient_<method>_atlas_01.png` and subsequent pages alongside its
individual figures, reusing the saved coordinates. Vocabulary atlases use a
separate family catalogue, count bars, and sparse numbered median anchors.

The censoring-aware figure retains the observed-minus-predicted calibration
residual in panel D. Negative values indicate over-prediction; positive values
indicate under-prediction. The supplied residual calculation, support mask,
bootstrap, and standardized color range are preserved. Maps and colorbars have
dedicated grid cells; status colors and observation shapes have separate keys.
Detailed residual diagnostics remain in the JSON output.

Binary, survival and joint finetuning export `plots/training_diagnostics_*.png`
and PDF from their Lightning CSV logs on rank zero. Curves are unsmoothed by
default, and validation-only metrics are included. No test data is evaluated
to produce these training diagnostics.

Evaluation favors survival discrimination when follow-up is available. If no
window is configured, the diagnostic display ends at the 90th percentile of
positive finite follow-up; this does not change the outcome definition. Cox
evaluation exports a ranking figure without probability calibration panels.
The gallery also includes the calibrated and injected-miscalibration examples
and a synthetic finetuning report.
