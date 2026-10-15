# Question-aligned Bayesian analysis

The current paper entry point is `python -m opera.run.bayesian_contrasts`.
The previous independent-error spline in `bayesian_rarity.py` and its runner
remain available for historical reproduction, not as the new primary analysis.
No real patient data or empirical Bayesian results are checked into this repo.

## What is fitted, and why

Fit a joint set of **independent positive-oriented paired contrasts** within one
scientific question family, metric, horizon and split. Derived contrasts are
computed in every posterior draw. They are not additional observations or
independent hypothesis tests.

| Family | Fitted basis | Derived questions |
|---|---|---|
| pooling | P1 disease pooling, P2 outcome pooling, P6 interaction | P3/P4 conditional increments; P5 combined benefit |
| routing | R1 router minus shared | Added value above full-panel shared learning |
| source_banks | R2 disease bank, R3 outcome bank, R7 interaction | R4/R5 conditional bank increments; R6 both banks |
| tabular | T2 shared minus XGBoost, T3 specialist minus XGBoost, R1 router increment, T4 XGBoost minus logistic | T1 router minus XGBoost |
| external | E1 direct external minus local, A1 DAPT minus direct, I1/I2 changes in routing increment | E2/A2 router effects; E3/E4 complete external package; I3 total routing change |
| external_tabular | T2, R1, E1, A1, I1, I2 | T5_shared/T5_router external+DAPT versus XGBoost |

For pooling, P3=P1+P6, P4=P2+P6 and P5=P1+P2+P6. Thus the two
paths from specialist to full panel agree exactly. For external pretraining,
E3=E1+A1 and E4=E2+A2. Joint fitting also avoids giving the same comparison
multiple votes by including its linear combinations as independent data.

Separate families use separate posteriors. Do not compare their posterior draws
as if they were one joint fit, or treat repeated R1 estimates as replications.
The external_tabular family is the explicit joint bridge when that claim is
needed. Six horizons are separate fits, not independent repeated evidence.
This is an association/heterogeneity analysis of training procedures; a rarity
slope is not a causal effect of withholding training labels.

## Model and uncertainty

For cell c and primitive contrast k:

`theta[c,k] = intercept[k] + trend(log2(1 + target training support))[c,k]
               + fine-disease effect[c,k] + outcome effect[c,k] + cell effect[c,k]`.

The default is a linear trend, with a slope measured per doubling of **1 plus**
training support. Zero training events are retained if a finite paired effect
is estimable. `trend: none` removes rarity regression; `trend: spline` is a
bounded sensitivity with explicit knots and Gaussian coefficient priors.
The spline is deliberately simpler than the legacy RW2 model. Its smoothness
comes from its fixed low-dimensional basis and coefficient prior. A shared
trend with varying intercepts is an assumption: it does not estimate a separate
rarity-response curve for each disease.

Gaussian cell residuals are analytically marginalized during NUTS, then drawn
from their exact conditional posterior for cell and finite-panel summaries.
This avoids sampling a large cell-effect funnel while retaining its uncertainty.
The likelihood is multivariate Normal within each fine disease, with covariance
estimated from the **same patient-bootstrap vectors** across outcomes, primitive
contrasts and fitted seeds. Fine diseases have disjoint evaluation patients,
so they form separate measurement-noise blocks. Shared random effects can still
link cells across diseases. Grade-paired/correlated endpoints keep their
measurement correlation. The bootstrap covariance uses fixed diagonal shrinkage
(default 0.02) and a tiny SE floor for positive definiteness. Record the block
condition numbers and examine shrinkage sensitivity (e.g. 0.10). This is an
estimated covariance approximation, not Bayesian inference over covariance
uncertainty or a guarantee that a Normal approximation fits every sparse cell.

Seeds are averaged **inside each bootstrap draw** before fitting. No independent
seed likelihood or `sigma_training` term is added. Intervals condition on the
specified fitted seed roster; the prediction analysis separately reports seed
range/SD. They do not quantify new pretraining seeds or the entire distribution
of future training runs. Missing seeds/contrasts, changing counts and mismatched
bootstrap row identities fail; nothing is silently intersected. At least 80%
and 100 jointly finite draws are needed per measurement block. Raw panel
intervals are suppressed if their jointly finite draws are insufficient.

The reference hierarchy shares across fine diseases, outcomes and cells.
Clinical parent nesting is **off by default**. `nest_diseases` and
`nest_outcomes` are independent sensitivities requiring explicit, consistent
cohort-group/family metadata. These priors express exchangeability; they do not
assert biological relatedness. Primitive contrasts have separate coefficients
and variance scales; unrelated gains are not forced toward a common intercept.

## Which population is being summarized?

`posterior_panel.csv` averages latent effects over the actual supported cell
panel. Default weights give equal weight to each fine disease, then each
available endpoint group within disease, then each outcome within that group.
This avoids giving grade-paired heads twice the group weight. `endpoint_group`
must come from the approved release and be present in the cell metadata fed to
the prediction analysis. Clinical families are separate from loss/reporting
endpoint groups. `equal_cell` is an explicit alternative sensitivity.

The target is the declared supported panel, not all possible cancer/outcome
pairs or a patient-volume-weighted population. The common-panel requirement can
restrict coverage for specialists or sparse AUC; freeze exclusions before
fitting and report the coverage inventory. Optional `cell_roster` points to an
explicit prespecified cohort/outcome CSV; its hash and counts are recorded.
Without it, the entire observed metric/horizon/split panel is required, including
cells seen only in other contrast families. Entirely missing family cells cannot
disappear through an implicit intersection. Probability-loss analyses can retain
single-class cells with valid bootstrap uncertainty.

`posterior_curve.csv` is the contrast-specific mean trend for a hypothetical
zero-random-effect task. It is **not** the finite-panel average or a predictive
interval for an individual task. The plots distinguish that curve from raw
paired points and posterior cell estimates. Curves interpolate only the
observed support range; uncertainty between cells is not sampling evidence of
a causal learning curve.

## Execution

First regenerate locked-prediction analysis outputs using the current runner.
Its manifest explicitly records patient draws shared across training seeds;
older manifests without that provenance or the executed contrast expressions
are rejected. Canonical contrast definitions and the model algebra are checked
together, so renamed/overridden comparisons cannot accidentally use the wrong
scientific interpretation. This reuses predictions
and requires no model training. Use an unsharded seed roster and one horizon for
the joint Bayesian job, or combine shards while preserving the same draw IDs.

Copy `opera/configs/bayesian_contrasts.example.yaml`, set secure paths, choose the
family/metric and lock priors plus practical tolerance. Values are deliberately
null in the template: AUROC, Brier and log loss do not have interchangeable
scales. Prior means are zero (neither benefit nor harm assumed). Set intercept,
random-effect and per-doubling slope scales using plausible gains and **prior
predictive** inspection, without using held-out effect magnitudes to tune them.
The pipeline checks the requested estimand against the upstream manifest;
known-status binary effects must not be described as full-risk-set IPCW effects.

```bash
python -m opera.run.bayesian_contrasts --config /secure/analysis/pooling.yaml --validate-only
python -m opera.run.bayesian_contrasts --config /secure/analysis/pooling_prior.yaml --prior-only
python -m opera.run.bayesian_contrasts --config /secure/analysis/pooling.yaml
```

`--prior-only` saves prior draws and panel summaries without fitting a posterior.
Review that output before committing the prior specification.

Use fresh output directories for each prespecified model/metric/horizon.
No training or tabular tuning is done by this command. Optional dependencies use the tested PyMC 5 / ArviZ 0 inference-data API; the
extra excludes the incompatible PyMC 6 major API. They
come from `pip install -e '.[bayesian]'`; smoke-tested runtime versions are
recorded in each model specification.

## Outputs and model checks

- `posterior_panel.csv`: weighted raw paired effects and Bayesian means/intervals;
  probability of any benefit, meaningful benefit, meaningful harm and practical
  equivalence at the declared tolerance. These are posterior probabilities,
  not multiplicity-adjusted discoveries.
- `posterior_cells.csv`: the same summaries per cell, raw paired intervals and
  audited panel weights. Partial pooling cannot manufacture data support;
  reportable-cell criteria remain the upstream analysis lock.
- `posterior_curve.csv`, `contrast_curves.pdf`, `contrast_panel.pdf`: question-
  aligned curves and forest-style panel comparisons, with diagnostic-only
  labeling if convergence fails. The band is posterior mean-trend uncertainty.
- `prior_predictive.nc`, `posterior.nc`, `predictive_checks.csv`: inspect plausible
  effect scales, prior tails, predictive coverage and covariance-whitened
  discrepancies. Tail fractions are descriptive diagnostics, not calibrated
  acceptance tests. Posterior in-sample checking is not held-out validation.
- `diagnostics.csv`, `model_specification.json`, `recipe.json`: divergences,
  R-hat, bulk/tail ESS, priors, tolerances, covariance regularization, contrast
  algebra, fit roster and input hashes. Require at least two chains, R-hat ≤1.01,
  ESS ≥400, zero divergences, no maximum-tree-depth hits and BFMI ≥0.3 by default. Missing diagnostics fail the gate.
- `model_inputs_INTERNAL.npz`, `model_cells.csv`: exact responses, shared draws,
  covariance blocks, regression design and cell roster for reproducibility.
  Keep these within the secure environment.

Convergence permits further model review; it is not automatic scientific
approval. Before primary reporting, compare linear/no-trend/spline fits,
flat/nested priors, sensible prior scales and covariance shrinkage, inspect
predictive diagnostics and raw-versus-shrunken estimates, and record sensitivity
of the principal conclusions. Keep raw paired results primary if conclusions
depend heavily on those assumptions. The production analysis still requires
real-data checks and the prespecified metric/coverage/multiplicity lock.

Implementation uses PyMC's [multivariate Normal likelihood](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.MvNormal.html)
and [posterior predictive sampling](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html).
