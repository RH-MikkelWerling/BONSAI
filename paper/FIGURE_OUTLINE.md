Current placement decisions: [FIGURE_PLACEMENT.md](FIGURE_PLACEMENT.md).
The fill-in Results contract is [RESULTS_OUTLINE.md](RESULTS_OUTLINE.md).
These supersede the older provisional main/supplement assignments below.

# Disease × outcome grid: figure outline

Rows represent fine diseases; columns represent clinical outcomes. A coloured
cell indicates supervision, not an adapter source. A highlighted target cell is
evaluated on the same patients under each scope. Each fit produces six horizons.

| Scope | Grid | Model fits needed | Scientific interpretation |
| --- | --- | --- | --- |
| S | One cell | One per estimable disease/outcome pair | Specialist reference |
| D | One column | One per outcome across applicable diseases | Disease pooling |
| O | One row | One per fine disease across applicable outcomes | Outcome pooling |
| J | Entire panel | One shared fit per seed | Both axes pooled |
| R | Entire panel, same as J | One locked readout fit per seed | Additional routing, not additional supervision |

Native artwork is `figures/supervision_grid.tex`. The separate, compilable
`supervision_grid_outline.tex` is a conceptual storyboard and contains no
empirical results. It is suitable for discussing the paper with a supervisor.

## Retained original figures

The original `information_axes.tex` grid and `modeling_paradigms.tex` remain
in the manuscript. The original empirical disease/outcome predictability atlas
and cohort-flow diagram also remain, alongside the paired effect displays.
The S/D/O/J grid is an additional model schematic, not a replacement for
either original grid. Figure numbering and main/supplement placement are
provisional until the complete artwork is assembled. The readout catalogue and
source-router diagram are retained with their exploratory/architectural status
explicit; representation analyses remain exploratory.

## Question-aligned empirical figures

**Additional schematic — supervision scopes (placement provisional).** The four grid panels,
prospective split and common initialization. Explain that specialists already
inherit the same pretrained representation. J and R occupy the same full grid;
the readout distinction appears as a small annotation rather than another
supervision scope. Table 1 is the approved cohort description, separate from
this schematic; do not reuse the historical table as final.

**Figure 2 — What is gained by each pooling axis?** At 365 days, aligned
fine-disease × outcome maps for D−S and O−S, then J−S and the factorial
interaction (J−D)−(O−S). Keep disease order, endpoint-group order and colour
scale fixed for comparable contrasts. Grey/hatching marks missing or
unsupported cells and never implies zero gain. Accompany maps with paired
support-stratified summaries; show uncertainty in aligned intervals, avoiding
unreadable error bars inside every tile. Include a compact R−J panel to show
whether routing adds to the ordinary shared model. If space requires splitting,
put the source-bank factorial in the supplement, not the pooling-axis effects.

Artifacts: per-cell contrast table with support and uncertainty; common paired
bootstrap draws; training support; seed-averaged effects. Question-aligned Bayesian
curves and finite-panel summaries use the new shared-covariance contrast model.
They appear beside raw points only after real-data validation; otherwise they
remain a descriptive supplement.

**Figure 3 — Does the neural approach improve on tabular prediction?** Aligned
S−X, J−X and R−X effects; paired aggregate intervals and calibration for the
main supported targets. Show X−logistic as the strength-of-baseline reference.
A predictability atlas can accompany these contrasts in the supplement; an
absolute AUROC map alone cannot answer comparative benefit.

Artifacts: frozen matched neural/XGBoost/logistic predictions, comparator
feature/tuning provenance, full-risk-set calibration outputs and paired effects.
Registry-enriched baselines must be visibly distinguished if inputs differ.

**Figure 4 — What does nation-scale pretraining add, and what does DAPT add?**
A three-column schematic (L, E, EA) crossed with two readouts (J, R), followed by
E−L, EA−E and EA−L paired estimates. Place scarcity/domain-mismatch strata below
the corresponding effects. Report routing-by-origin interactions without
reselecting the router on nation-scale-model held-out data. Explicitly show that EA starts
from the same remapped E artifact. Do not imply matched model size if nation-scale encoder
architecture differs.

Artifacts: six matched J/R origin arms, encoder/transfer/DAPT hashes, same-seed
initialization provenance, transfer overlap report, paired contrasts and costs.

## Required supplement

- Complete six-horizon effect atlas and estimability/coverage inventory.
- Architecture-matched N/RD/RO/RDO bank contrasts and tuning-only rank/query
  selection evidence, including null findings.
- Calibration, Brier/log loss and full-risk-set censoring sensitivities.
- Raw versus hierarchical scarcity summaries, prior/model checks and hierarchy
  sensitivity; shared-draw uncertainty provenance.
- Tuning-defined negative-transfer strata, patient-history/covariate gains,
  calendar/domain-mismatch patterns and gate/source-use descriptions.
- Delayed-diagnosis coding audit and same-fit sensitivity.
- Compute/parameter/memory tradeoffs, seed variability, endpoint dictionary,
  cohort flow, approved Table 1 and release provenance.

Embeddings and source-mechanism interventions are optional. Do not let a visually
appealing representation map displace a contrast answering a headline question.
Every empirical display must ship its tidy source data and plotting recipe;
patient-level artifacts remain on the secure environment. These titles express
questions, not assumptions that the joint or routed model wins.

## Figure prototypes (design proofs, atlas style)

`python -m opera.run.render_paper_figure_prototypes --output-dir paper/render/prototypes`
renders every data figure from synthetic tables with the same columns as the
analysis outputs; each artifact is watermarked SYNTHETIC DESIGN PROOF. The
functions live in `opera/visualization/paper_figures.py`, so the same code
renders the real figures from the locked tables.

| Prototype | Paper role | Real input |
| --- | --- | --- |
| `fig_atlas` | Predictability atlas (main) | joint-model held-out IPCW AUC per outcome × horizon + training events |
| `fig_pooling` | Q1: disease × endpoint-group maps (median gain per tile) for D−S, O−S, J−S and interaction; gain vs training events; by endpoint group | `contrasts.csv` (P1, P2, P5, P6) + cell metadata; Bayesian `posterior_curve.csv` |
| `supp_pooling_cells` | The same four contrasts at full disease × outcome resolution | `contrasts.csv` |
| `fig_tabular` | Q2: S/J/R vs XGBoost (AUC, Brier), support curve, horizons, XGBoost vs logistic | `contrasts.csv` (T1–T4) + curve |
| `fig_external` | Q3: L/E/EA × J/R absolute; E−L, EA−E, EA−L; by disease group | `estimates.csv`, `contrasts.csv` (E1–E4, A1–A2) |
| `supp_routing` | Routed vs joint; adapter-bank factorial with N≈J equivalence margin | `contrasts.csv` (R1), `router_null_equivalence.csv`, effect summaries (R2–R7) |
| `supp_router_usage` | Which patients use which sources: disease-bank borrowing (home hatched), outcome-group borrowing, which bank carries the correction by disease × target group, and how much weight leaves the shared path | analysis `route_weights` table (`route_summaries`, needs `--export-routes`) |
| `supp_horizons` | Pooling effects across six horizons on a shared scale | `contrasts.csv`, all horizons |
| `supp_calibration_maps` | IPCW kernel risk and local calibration for several outcomes on the joint model's single patient embedding | 2-D embedding coordinates + held-out time/event + predicted CIF per outcome |

Conventions: lettered left-aligned panel titles; diamonds with bars for
summaries (IQR across cells, or 95% CI for panel contrasts — stated on the
axis); log-scaled training events; indigo-to-gold horizon ramp; centred
diverging scale with grey for cells that could not be estimated (never zero);
no figure titles or footer prose — interpretation goes in the caption.
