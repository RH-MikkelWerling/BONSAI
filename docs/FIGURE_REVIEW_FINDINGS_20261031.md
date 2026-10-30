# Figure review findings, 31 October 2026 (night)

Independent check of the 86 live figures in the 30 October revision against the
author's notes, the implemented-change log and `FIGURE_STYLE.md`. Three review
passes over the revised PNGs (and the originals where a note compared them).
Synthetic layout checks cannot show data-specific problems; clinical plates in
particular are one template rendered per index.

## Fixed the same night

- **OPERA colour identity.** `MODEL_STYLE` gave Joint the OPERA indigo and OPERA
  lavender, against the style module. Swapped (OPERA and routed indigo, Joint
  lavender), also in the subgroup contrasts. Affects every comparison figure.
- **Embedding galleries.** The author asked to keep the originals (one point per
  patient, final-model predicted risk, 0–1). The revision had switched the
  default to smoothed observed-risk surfaces, which also produced stray squares
  and holes. Default restored to `predicted_points`.

## Still to fix

### Author requests not implemented or regressed
- `fig_calibration`, `fig_benchmarks` D: reliability examples degenerate (one or
  two bins). Neither the prespecified rule nor the fallback list picks AITL, so
  this is a local-redraw artefact; verify the examples in the server refresh.
- `supp_national_where`: the "Overall" row the tracker claims is not drawn.
- `supp_calibration_maps` / `supp_router_arrows`: arrows requested both in their
  own panel and overlaid on the maps; only the separate row exists. Maps shrank
  with large gutters (regression against "keep spacing/panel size").
- `supp_embedding_comparison`: original pre-LDH layout preferred; revision is a
  five-column point layout; IPI contrast still weak.
- `supp_attribution_structure`: panel B column headers still collide (the one
  requested fix); A/B titles misaligned; subtitle in A.
- `supp_recent_record_gaps`: only the 1-year page exists; the six horizon and
  clinical-family pages in the change log are missing; 14/21-day colours close.
- Clinical-score plates (all indices): legend and three prose lines under A
  still awkward; threshold "≥ k" labels lost; B legend floats between B and C.
- `supp_score_overview` 1–3: no legends in A/B; KM line styles and category
  colours inconsistent with the plates; cryptic "M:/I:" labels; empty gutter.
- `router_mechanism_probability`: author suggested pairing with the AUC plate.
- `fig_headline`: mixed mean/CI and median/IQR rows not distinguished; disease-
  pooled and XGBoost share a colour; D legend far from panel.

### Visible bugs
- `fig_attribution`, `supp_attribution_atlas`: about eight all-grey concept
  columns (upper ALT … potassium bins, systemic glucocorticoids).
- `supp_attribution_patient`: bubble values (+0.54/+0.78) disagree with
  `fig_attribution` D (+6.7/+9.8 pp) for the same events; no size/colour key.
- `fig_patient_map`: stray squares above maps and holes in C/D (same surface
  issue as the gallery); E title misaligns row two.
- `proposed_support_sensitivity` B: two identical series overplotted.
- `information_axes` D: a cohort dot sits on the word "broader".
- `fig_benchmarks` C: panel letter collides with the y-axis label.

### Style (shared rules)
- Footer prose and subtitles remain on many figures (mechanism family, clinical
  plates, borrowing maps with provenance hashes, regions, strategy, support,
  source_router schematic); move to captions.
- Clinical plates: panel letters not bold; font differs from DejaVu Sans; one
  ordinal palette across all indices (Intermediate is sienna in FLIPI2, gold in
  CLL-IPI); age bins differ between the score and age plates.
- `supp_regions`: hatching means "CI includes 0", clashing with hatched =
  unavailable elsewhere; two comparisons share indigo; label halos.
- `supp_support`: grey used for zero events (grey must mean unavailable).
- `fig_external` B: decorative fill between curves; floating prose.

### Router and borrowing figures (redesign for the selected router)
`fig_hierarchy`, `supp_routing`, `supp_router_usage`, `fig_borrowing_map(_zero)`,
`supp_borrowing_detail(_zero)` and the `source_router` schematic are redrawn
for whichever router is selected. New renderers in
`opera/visualization/pooling_router_figures.py` (`figure_how_to_borrow`,
`figure_outcome_routes`) read the mechanism audit outputs; tested on synthetic
audit-format data. They still need the R_PT aliases and router-neutral labels.
