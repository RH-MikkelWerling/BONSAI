# TRIPOD+AI reporting crosswalk — working draft

Use the [official expanded checklist](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf)
for the final numbered submission form. This topic crosswalk identifies evidence
and locations; it is not a declaration that all checklist items are satisfied.

| Reporting topic | Planned manuscript location | Still required |
| --- | --- | --- |
| Study purpose, model development/evaluation and intended use | Abstract / Introduction | Actual results; intended users and clinical setting stated precisely |
| Data sources, dates, participants and sample size | Methods / Table 1 / definitions supplement | Frozen extraction dates, supervisor-approved counts and sample-size rationale |
| Outcomes, predictors and measurement timing | Methods / definitions supplement | Close clinical/source questions and prediction-time availability audit |
| Missing data and eligibility | Methods / risk-set and feature tables | Missingness by split, exact imputation/training-only fitting and exclusion counts |
| Preprocessing, architecture, training and tuning | Methods / model-reporting supplement | Completed configs, selection history, seeds, actual budgets and provenance |
| Validation and performance estimation | Methods / Results / metric tables | Primary estimand/weighting/threshold locks; uncertainty and follow-up diagnostics |
| Full model performance, calibration and subgroup evaluation | Main figures / supplement | Real results and unsupported-cell status, no synthetic values |
| Model specification and access | Model tables / Code availability | Versioned code/configs, model access conditions and reproduction commands |
| Limitations, applicability and interpretation | Discussion | Match conclusions to final effects; distinguish temporal evaluation and clinical impact |
| Ethics, funding, contributions and patient/public involvement | Methods / end matter | Verify approval/waiver details, funding roles, contributions, and involvement or its absence |

For each final official item, record the exact page/section and any justified
not-applicable response. Do not invent patient involvement, registration,
external validation or completed clinical deployment.
