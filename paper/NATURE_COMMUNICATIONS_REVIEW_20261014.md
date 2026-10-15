# Full manuscript pass — 14 October 2026

The potential advance is an empirical answer to how clinical experience should
be shared when prediction targets are individually sparse. The study separates
supervision across diseases, supervision across outcomes, and nation-scale
representations. The architecture enables those experiments; the generalisable
finding must come from the contrasts. Synthetic figures provide no evidence of
performance, clinical benefit or likely acceptance.

## Narrative and corrections

The clinical stakes are threefold: reliable prediction for sparse targets;
justification for changing the practical tabular modelling approach; and
justification for investment in reusable population encoders. The Introduction
now explicitly links these decisions. The Results transitions explain why a
pooling gain does not establish superiority over XGBoost, why direct transfer
and adaptation need separate comparisons, and why panel metrics do not replace
clinical interpretation. Routing is subordinate to supervision, not a fourth
headline question. Embedding maps remain descriptive.

Corrected the first-line prose count to agree with the draft flow (37,557;
supervised 37,531). Counts still await approval. Changed “prospective” to
retrospective temporal evaluation. The PHAIR cutoff is a requirement pending
provenance, not an established leakage-control fact. Primary metric/weighting
locks remain explicit. Removed Discussion mini-headings without dropping the
implications or limitations. Revised disclosure wording to accommodate patient
embedding plots: individual-point release needs applicable disclosure review.

## Evidence needed for a strong submission

| Priority | Evidence | Acquisition / permitted interpretation |
| --- | --- | --- |
| Essential | Separate disease/outcome/combined/conditional effects and interaction | Matched current-256 S/D/O/J runs, common target rosters, paired bootstrap uncertainty and seed summaries. Establishes supervision effects; not biological mechanism. |
| Essential | Strong practical baseline | Complete XGBoost/logistic feature matrices, training-fitted preprocessing, audited information availability and tuning budget, matched test risk sets. |
| Essential | Nation-scale reuse and domain adaptation | PHAIR corpus/cutoff/vocabulary audit; identical mapped E starting artifact for EA; fixed downstream J/R protocols and E−L, EA−E, EA−L. A package comparison does not isolate scale alone. |
| Essential | Accurate probabilities and supported horizons | AUC, Brier and calibration; follow-up/event counts; IPCW weight/positivity diagnostics and relevant sensitivity. Two-year plots need actual follow-up support. |
| Essential | Clinical validity and cohort accounting | Approved fresh Table 1/flow; treatment-line validation evidence where available; endpoint versions/deviations; diagnosis-at-index audit and same-fit late-code sensitivity. Future lab ascertainment may condition the evaluated population and needs explicit interpretation. |
| Essential | Full numeric evidence | Absolute performance and paired contrast tables with exact denominators, unavailable fits, harmful cells, all horizons and seed variation. Figures do not replace reusable tables. |
| Essential | Reproducibility/governance | Configurations, checkpoint hashes, actual parameter/step/compute counts, access procedure, archive/license/environment, figure source data, applicable reporting summary and completed reporting items. |
| Strong addition | Robustness beyond one average | Disease/outcome heterogeneity, calendar stability, sparse-target coverage and tuning-selected portfolio, within the existing direction. |
| Claim-dependent | Independent geographic validation | Valuable if feasible; temporal Danish validation cannot establish transport. A held-out region requires a valid design and potentially new fitting. |
| Claim-dependent | Label-efficiency mechanism | Controlled training-label subsampling on prespecified tasks if claiming sharing reduces required labels. Natural support associations alone cannot prove it. |
| Claim-dependent | Pretraining versus none or pure scale | Random-initialized matched control for the former; architecture/objective/budget-matched scale controls for the latter. Neither is answered by L/E/EA alone. |
| Optional | Embedding/vocabulary figures | Descriptive supplement with train-fitted projections, index-time inputs, metadata alignment, checkpoint provenance and disclosure review. |

My assessment: Nature Communications is a defensible ambitious target if a
substantial, reproducible conclusion survives strong baselines and these
validity checks. A selective benefit for sparse targets can be more informative
than a large opaque average gain. A null router can support choosing less
complexity. A null nation-scale result can be useful, but the main advance must
then be carried by the supervision findings. No additional architecture is
needed merely to increase the experiment count.

## Requirements and precedents checked

The live [Article guidance](https://www.nature.com/ncomms/submit/article)
recommends at most 5,000 main-text words excluding abstract/methods/references/
legends, an abstract of at most 200 words, up to ten figures/tables depending
on length, and a succinct Discussion without subheadings. Initial formatting
is flexible. Its suggested order is Introduction–Results–Discussion–Methods;
our coauthor version retains Methods before Results and can be reordered in a
venue-specific export. The older
[submission guide](https://www.nature.com/documents/ncomms-submission-guide.pdf)
is dated 2019; prefer the live Article guidance for differing numerical limits.
Use the journal's [resources](https://www.nature.com/ncomms/submit/resources)
for the current reporting summary and computational reporting requirements.
A TRIPOD topic map is not completed compliance.

[TransformEHR](https://www.nature.com/articles/s41467-023-43715-z) is a relevant
Nature Communications precedent for pretraining with downstream clinical tasks,
common/uncommon target analysis and out-of-domain evaluation; its exact
experiments are not acceptance requirements. The
[multi-centre adaptation study](https://www.nature.com/articles/s41746-024-01166-w)
provides related evidence on shared encoders and continued pretraining.
Our distinguishing emphasis is the separated disease/outcome supervision
contrasts and their relationship to nation-scale reuse, not generic transformer
novelty.

## Synthetic layout settled in this pass

Nine main figures plus Table 1: modelling paradigms; original information axes;
cohort; predictability; pooling; tabular comparison; calibration; nation-scale
transfer/adaptation; clinical risk groups. Calibration is promoted to main
because probability quality is necessary to judge risk prediction. Routing
and the tuning-selected portfolio move to supplementary figures, with their
main Results text retained. Neither is hidden or removed. All nineteen
prototype result plates remain included, alongside all original schematics,
full R-derived definitions and the model/training supplement.

Keep plot schemas, palette, order and figure roles stable now. Real outputs
must still determine ranges, intervals and final captions; synthetic data cannot
freeze those. The five-model embedding matrix is supplementary. A possible
later compact submission could combine selected plates, but must recompute
panel letters and preserve readability rather than stack duplicate A/B panels.
