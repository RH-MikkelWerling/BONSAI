# Paper production plan

This file separates material that can be locked while model runs continue from
material that depends on final held-out predictions. Every empirical figure
should be generated from a tidy source table and exported as PDF/SVG plus its
underlying CSV or Parquet artifact.

## Can be finalized now

1. **Cohort definition and flow**
   - Unique source, linked, history-eligible and temporally split patient counts.
   - Exclusion reasons with mutually exclusive counts.
   - Outcome-specific risk-set counts in a supplementary table, not crowded into
     the main flow diagram.
2. **Table 1**
   - Demographics and observation-history characteristics by temporal split.
   - Disease groups generated from the locked cohort manifest.
   - Use one record per patient at the shared index date.
3. **Endpoint catalogue**
   - Outcome name, clinical domain, eligibility, competing event, event counts,
     follow-up and estimable horizons.
   - Keep clinical reporting domains separate from optimization weights/trunks.
4. **Study design and model schematics**
   - The disease–outcome supervision grid and separate figure-outline handout.
   - Exact temporal split, leakage controls, pretraining windows and comparator
     definitions in Methods.
5. **Supplement scaffolding**
   - Cohort definitions, code families, outcome definitions, missingness,
     hyperparameters, compute, and model-selection rules.

The original information grid, modelling paradigms, cohort flow and empirical
predictability atlas are retained. The new supervision schematic supplements
them. Numeric figure labels below are a question storyboard, not final numbering.

## Requires locked model outputs

| Display | Main question | Required artifact |
|---|---|---|
| Additional supervision schematic (placement provisional) | What changes across S/D/O/J? | Native TikZ, approved scope/split definitions; no performance numbers |
| Fig. 2 Pooling and routing increments | Which pooling axis helps, and where? | P1–P6 and R−J paired effects, support, shared bootstrap draws, seed summaries |
| Fig. 3 Tabular comparison | Does the neural pipeline improve on standard tabular models? | Patient-matched S/J/R/X/logistic predictions and calibration |
| Fig. 4 External transfer and adaptation | What does external pretraining add; what does DAPT add? | L/E/EA × J/R fixed-protocol predictions, provenance, costs |
| Required supplement | How robust and broadly applicable are the findings? | Six horizons, coverage, bank controls, calibration, strata, coding audit and statistical sensitivities |

The complete panel/storyboard contract is in [FIGURE_OUTLINE.md](FIGURE_OUTLINE.md).
The question hierarchy is in [the full analysis plan](../docs/full_analysis_plan_20261014.md).
Embedding maps are optional supplements. Hierarchical rarity curves require
model/dependence validation and must accompany raw effects. Neither replaces
the headline contrasts.

## Locked visual rules

- Main text emphasizes 365 days; all six horizons appear in supplements.
- Main summaries show both the average paired effect and heterogeneity, never
  only a macro average.
- Uncertainty intervals and event support accompany performance estimates.
- Unsupported cells remain visible as `not estimable`.
- Clinical families organize reporting only unless an analysis explicitly tests
  family-based sharing or weighting.
- No interactive-only figure is required for the paper.
