# Claims and experiments

The potential change is how prediction is developed: shared representations and
supervision could support multiple specialised risks while each clinical use
retains its own disease-, outcome- and horizon-specific evaluation. This remains
a hypothesis until the locked results are available.

Clinicians should see the potential for a reliable profile of disease course
and complications. Machine-learning readers should see a factorial attribution
of supervision breadth, selective adaptation and representation reuse. General
readers should understand why experience divided among small prediction tasks
might be used more effectively together. These are three readings of the same
study, not three different sets of claims.

| Question | Evidence | Permitted interpretation / boundary |
| --- | --- | --- |
| Do other diseases help? | D−S, J−O | Effect of cross-disease supervision under this protocol; not biological interchangeability or a pure sample-size effect. |
| Do other outcomes help? | O−S, J−D | Effect of cross-outcome supervision; distinct from an outcome-adapter effect. |
| Do the axes complement one another? | J−S, J−D−O+S | Joint effect and interaction on the specified metric scale; no guarantee of benefit in every cell. |
| Does routing add value? | R−J; matched bank controls | Incremental adaptation benefit; gate weights do not establish clinical mechanisms. |
| Is one model a practical alternative? | J versus tuning-selected S/D/O; measured resources | Accuracy/coverage/resource trade-off against this portfolio; fit count alone does not establish efficiency. |
| Does the pipeline beat practical baselines? | S/J/R versus tuned XGBoost/logistic | Superiority on declared metrics and matched targets; input/tuning parity matters, and sequence ordering alone is not isolated. |
| Does nation-scale experience transfer? | E−L with fixed downstream protocols | Delivered PHAIR procedure versus delivered local procedure, conditional on valid provenance; not pure scale when other factors differ. |
| Does local adaptation help? | EA−E from identical mapped E | Increment attributable to the specified continued-pretraining procedure; EA−L alone does not separate the stages. |
| Is risk clinically interpretable? | Matched clinical-score populations; locked groups and survival/CIF estimates | Observed risk stratification, with missingness/support; not demonstrated clinical benefit or treatment selection. |
| What do representations look like? | Matched five-stage embedding maps | Descriptive projection structure; no performance or biological mechanism claim. |

If both supervision axes help, the results support developing related clinical
risks together. If only one helps, they identify more selective sharing. Useful
predictions where specialists cannot be fitted establish coverage, not a
measured gain over an unavailable specialist. If PHAIR improves prediction,
population representations become a credible reusable component; EA−E describes
the local work still needed. Strong tabular comparisons and calibration determine
whether these possibilities justify changing the practical approach.

The headline follows real findings, including null and harmful contrasts.
Reduced label requirements need controlled label subsampling; geographic
transport needs independent testing; improved care needs action-specific and
prospective impact evidence beyond this study.

Clinical-score comparisons must include original intended-endpoint benchmarks
against tabular and sequence models, separately from treatment-complication
extensions. See CLINICAL_SCORE_ENDPOINT_BENCHMARKS.md. Raw score rankings permit
discrimination comparisons; probability metrics require a justified, locked
mapping. Common score-available patients are mandatory for paired comparisons.
