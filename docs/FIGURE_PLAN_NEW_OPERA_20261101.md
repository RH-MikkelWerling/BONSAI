# Figures for the new OPERA (outcome-specific reading + own corrections)

1 November 2026. If a stage-9/13 design is selected, OPERA's predictions come
from it (`OPERA_PAPER_OPERA_MODEL` / `OPERA_PAPER_ROUTED_MODEL`), so every
prediction-based figure only needs the new alias. Figures that showed the
router's internals (routes, sources, borrowing between tasks) have no
counterpart in the new design and are retired; new figures show what the new
design can say: what is shared, where each task specialises, and why.

## A. Swap the model only (no design change)

| Figure | Where | Note |
|---|---|---|
| fig_headline (OPERA vs reference methods) | main | |
| fig_benchmarks / tabular | main | the XGBoost gap by outcome group becomes a panel (see C4) |
| fig_risk_groups (vs NCCN-IPI) | main | |
| fig_external (national pretraining) | main | "routed model" → OPERA; national R fits carry the new readout |
| fig_calibration, supp calibration | supp | |
| fig_atlas, atlas matrix | supp | |
| clinical score plates, score overview, score age | supp | |
| who benefits, regions, coverage, support, support sensitivity, horizons | supp | |
| supp_rescue (cells where joint learning hurt) | supp | "routed" → OPERA |
| supp_strategy | supp | OPERA in the expanded candidate set |
| attribution figures (atlas, structure, patient, recent-record gaps) | supp | pipeline now supports the new readout (diagnosis index, randomisation test); rerun |

## B. Caption or panel change (same figure, different meaning)

| Figure | Change |
|---|---|
| fig_paradigms (top: modelling paradigms) | panel C: "a sequence model shares one representation across outcomes" → shares the encoder, each outcome reads the record through its own query, small per-diagnosis and per-outcome corrections |
| fig_patient_map, embedding gallery, embedding comparison | the map is the **shared** representation (before outcome-specific reading); say so. Predicted-risk panels use the final model and stay |
| supp calibration maps | drop the "projected routed corrections" row; keep event, observed, predicted, residual |

## C. Retire (router internals) and replace (what the new design can show)

| Retired | Why | Replacement |
|---|---|---|
| fig_source_router (schematic) | describes routing | **C1. OPERA schematic**: shared encoder → per-outcome query attention (gated onto the shared summary) → own diagnosis and outcome corrections (+ global) → per-outcome head |
| fig_hierarchy (own vs borrowed corrections by support) | borrowed corrections no longer exist | **C2. Where the gain comes from** (main text): J → + own corrections → + outcome-specific reading, by training support and by outcome group; synergy; ablations (no diagnosis / no outcome correction, stage 12) |
| supp_routing (routed vs joint, adapter banks) | router comparisons | **C3. The design search** (supp): every design (stages 3–13) against J and R_AB with paired intervals, grouped: routing variants, structure, capacity, reading, reading + corrections, shrinkage/cross; the encoder-rate sensitivity |
| supp_router_usage (route weights) | no routes | **C5. Specialisation per outcome** (supp): attention gate |tanh g| and own-correction size per outcome, against events and by endpoint family (does the model specialise where data allow?) |
| fig_borrowing_map, supp_borrowing_detail (source removal) | no sources | **C6. Outcome reading map** (supp): clustered similarity of outcome queries, coloured by endpoint family (which outcomes read the record alike) |
| supp_router_arrows | routed corrections | none needed; C2/C5 carry the interpretation |
| — | new | **C4. Where XGBoost still leads** (supp or benchmarks panel): gap by outcome group (liver/coagulation, electrolytes, metabolic vs blood counts), with OPERA's gain over R_AB in the same groups; the value-resolution explanation |
| — | optional | **C7. What outcomes read** (supp): per-head attention by concept group and time window (needs an aggregate export; one day) |

## D. The pooling ladder (fig_pooling, main)

Extend the existing S / D / O / J figure with the partial-pooling rung:
specialists (no pooling) → pooling one axis (D, O) → complete pooling (J) →
partial pooling (OPERA). It is the paper's spine and replaces the router as the
organising picture.

## Data each new figure needs

| Figure | Inputs |
|---|---|
| C2 | paired contrasts (analysis output): R_PF_OFF−J, J_OA−J, OPERA−J, ablations; cell support |
| C3 | paired contrasts for all designs (tuning: development; test: selected only) |
| C4 | paired contrasts T1/T6 by outcome group |
| C5 | selected checkpoint (gates, adapter norms) + training events per outcome |
| C6 | selected checkpoint (outcome queries) + endpoint families |
| C1 | TikZ, thesis style |
