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
   - The three TikZ figures in `figures/`.
   - Exact temporal split, leakage controls, pretraining windows and comparator
     definitions in Methods.
5. **Supplement scaffolding**
   - Cohort definitions, code families, outcome definitions, missingness,
     hyperparameters, compute, and model-selection rules.

## Requires locked model outputs

| Display | Main question | Required artifact |
|---|---|---|
| Fig. 4 Predictability atlas | What can be predicted? | One row per model × cohort × outcome × horizon × metric, with uncertainty and support |
| Fig. 5 Scale and scarcity | When does transfer help? | Paired predictions/contrasts plus cohort size and event support |
| Fig. 6 Model comparison | Does OPERA beat XGBoost? | Patient-matched OPERA and XGBoost predictions on held-out data |
| Fig. 7 Representation | What structure was learned? | Fixed embeddings, projection coordinates and held-out labels |
| Architecture supplement | Which readout wins and where? | Tuning-only architecture metrics plus selection provenance |

## Locked visual rules

- Main text emphasizes 365 days; all six horizons appear in supplements.
- Main summaries show both the average paired effect and heterogeneity, never
  only a macro average.
- Uncertainty intervals and event support accompany performance estimates.
- Unsupported cells remain visible as `not estimable`.
- Clinical families organize reporting only unless an analysis explicitly tests
  family-based sharing or weighting.
- No interactive-only figure is required for the paper.
