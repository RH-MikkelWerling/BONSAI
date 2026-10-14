# Proposal: estimating router support shrinkage from training data

Recorded for methodological follow-up rather than expanding the current finalization experiments.

OPERA currently scales each source correction by `events / (events + tau)` with tau=10 in guarded runs. This is a source-support multiplier, not itself a posterior estimate of how much one task should borrow from another. The router already learns task-dependent mixture weights.

There are relevant precedents. Adaptive historical borrowing models infer commensurability between historical and current data and reduce borrowing when those sources conflict: [Hobbs et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3134568/). In Bühlmann credibility, `Z = n / (n + K)` with `K = expected process variance / variance of hypothetical means`: [Society of Actuaries study note](https://www.soa.org/globalassets/assets/files/edu/c-21-01.pdf). The formal resemblance motivates a proposal; it does not make the current neural multiplier a credibility estimator.

Before estimating tau, specify the quantity being pooled, its sampling uncertainty, and the heterogeneity across comparable sources. Disease event counts pool correlated outcomes; outcome event counts have different eligibility, prevalence, grades, and censoring. Raw variances of outcome prevalence across different endpoints do not provide a valid credibility constant for adapter quality.

Making tau trainable through prediction loss alone also introduces a scale tradeoff: for positive shrinkage, an adapter's up matrix can compensate for a smaller multiplier. Learned tau is therefore confounded with adapter magnitude, gate weights, and regularization; it should not automatically be interpreted as estimated borrowing reliability. Appropriate priors/constraints or a separately estimated uncertainty model would be needed for that interpretation.

A future design should distinguish source-estimation uncertainty from target-source compatibility. Candidate approaches are a training-only hierarchical model for comparable correction/effect estimates, or a validated reliability estimate from training resampling, combined with the learned router. Any calibration/selection must remain within train/tuning, with held-out sealed. This proposal is documented locally; no remote issue has been posted.
