# Introduction citation audit — 14 October 2026

Check the source's actual claim, study population, task and publication metadata
before citing it. A paper's existence or title alone does not establish a broader
claim. Hypotheses and our experimental design are labelled as such.

## New or corrected claims in this pass

| Claim / citation | Source checked | Scope and action |
| --- | --- | --- |
| Individually uncommon lymphoid diseases / Sant 2010 | [HAEMACARE original incidence-study abstract and metadata](https://pubmed.ncbi.nlm.nih.gov/20664057/) | Supports subtype incidence variation; not a contemporary Danish count or a claim that every disease/outcome is rare. Added citation. |
| Expected benefit from shared task signals / Caruana 1997 | [Original paper, abstract and Sections 1.3–1.4 and 7.7](https://aima.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/caruana-1997.pdf) | Shared representations can improve generalisation; the paper also discusses harm. Added as rationale, not proof for OPERA. |
| Rare-phenotype gains / Ding 2019 | [Original PSB paper, abstract and conclusions](https://psb.stanford.edu/psb-online/proceedings/psb19/ding.pdf) | Rare phenotypes benefited; common ones could worsen. Retrospective phenotyping, not prospective treatment-start outcomes. Added with this distinction. |
| Infection/treatment-need prediction / Agius 2020 | [Original paper and reported history-ablation analyses](https://www.nature.com/articles/s41467-019-14225-8) | CLL-TIM predicts infection or treatment within two years of diagnosis; histories contribute. Does not establish cross-disease benefit or post-treatment utility. |
| Agius 2022 | [Publication record](https://pubmed.ncbi.nlm.nih.gov/34612160/) | Recommendations article, not the primary evidence for the preceding prediction claim. Removed from that citation cluster. |
| Multi-task negative transfer / McDermott 2020 | [Original preprint abstract](https://arxiv.org/abs/2007.10185) | Joint training can harm; multi-task pretraining with single-task fine-tuning can help. Not a guarantee for our architecture. |
| Multi-task EHR representations / Harutyunyan 2019 | [Original benchmark paper](https://www.nature.com/articles/s41597-019-0103-9) | Supports the existence and evaluation of shared clinical-task representations; not universal superiority. |
| Diagnostic scores | [NCCN-IPI](https://pubmed.ncbi.nlm.nih.gov/24264230/), [CLL-IPI](https://pubmed.ncbi.nlm.nih.gov/27185642/), [R-ISS](https://ascopubs.org/doi/abs/10.1200/JCO.2015.61.2267) | Checked original-score metadata/abstracts; intended disease populations remain explicit. |
| CORE-BEHRT | [Original preprint](https://arxiv.org/abs/2404.15201) | Optimises representation, architecture and training with multi-task evaluation; no performance numbers imported into this manuscript. |
| Med-BERT | [Original publisher paper](https://doi.org/10.1038/s41746-021-00455-y), [publication record](https://pubmed.ncbi.nlm.nih.gov/34017034/) | Publisher direct-open sometimes redirects; original indexed content supports structured-EHR contextual pretraining. |

BEHRT, MOTOR, EHRSHOT, EveryQuery v3 and the earlier DLBCL paper were checked
against their original papers in the preceding introduction passes. Their
restricted claims and links are retained in INTRODUCTION_ARGUMENT.md and the
bibliography. The Danish registry/resource papers support linkage infrastructure;
they are not cited as proof that every endpoint has complete ascertainment.

The expectation that sharing improves OPERA remains the study hypothesis. No
literature citation can substitute for measuring its matched pooling effects.
The outcome-support distribution must come from our approved release; no cited
phenotyping or incidence study supplies those counts. This audit records claim
checks, not a completed systematic literature review or a bibliography-wide audit.

## Model overview and national-pretraining expansion

The CLL-TIM and DLBCL examples were rechecked in their original papers before
expanding the model overview. Their published studies support research use of
clinical/laboratory histories and gradient-boosted prediction, not a claim of
routine clinical deployment. The sequence-model description is linked to
BEHRT/Med-BERT and Figure `fig:paradigms`; no universal neural superiority is
claimed. L/E/EA describe this study's procedures rather than published results.
Population pretraining, direct supervised transfer and local continued SSL are
separate steps. Exact external population coverage, dates, architecture and
budget remain checkpoint/provenance reporting requirements. Neither shared
national registers nor a successful transfer establishes geographic portability.

## Meta-learning evidence

MetaPred's original KDD paper (2019, pp. 2487–2495,
10.1145/3292500.3330779) was checked against the author-hosted PDF and original
publication record. It learns from related clinical prediction tasks and adapts
to low-resource targets; it supports borrowing as a rationale, not a claim that
OPERA implements meta-learning. Time Associated Meta Learning was checked at
https://arxiv.org/abs/2303.02570 (author list, abstract and preprint status); its
sharing across related events/horizons is described without importing numerical
gains or claiming prospective haematology validation. Neither is a direct
benchmark result for the current treatment-start panel.

## Introduction and Methods claim-level pass

New Methods citations were checked against original publication pages, papers
or official documentation before insertion:

| Method / key | Checked source | What it supports; what it does not |
| --- | --- | --- |
| Transformer / vaswani2017attention | [NeurIPS original publication](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | Transformer architecture precedent, not our exact causal encoder or hyperparameters. |
| Time2Vec / kazemi2019time2vec | [Original preprint](https://arxiv.org/abs/1907.05321) | Learned vector representation of time; our scaled variant is explicitly an adaptation. |
| Interval hazards / gensheimer2019nnetsurvival | [Original publication record/abstract](https://pubmed.ncbi.nlm.nih.gov/30701130/) | Neural discrete-time survival precedent; does not define our competing-risk horizon-BCE objective. |
| Competing-risk prediction / lee2018deephit | [Original AAAI article](https://ojs.aaai.org/index.php/AAAI/article/view/11842) | Joint time/cause prediction precedent. DeepHit learns a distribution directly; do not describe our implementation as identical. |
| Gated sharing / ma2018mmoe | [Authors' publication page](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) | Multi-gate shared experts; not evidence of clinical outcome benefit or our anchored update rule. |
| Adapters / houlsby2019adapters and pfeiffer2021adapterfusion | [ICML](https://proceedings.mlr.press/v97/houlsby19a.html), [ACL](https://aclanthology.org/2021.eacl-main.39/) | Adapter and task-composition precedents; source ownership/detachment are our implementation. |
| XGBoost / chen2016xgboost | [Original author preprint with published DOI](https://arxiv.org/abs/1603.02754) | Algorithm and correct 2016 authorship (Chen and Guestrin). Actual feature/tuning recipe is study-specific. |
| KM / kaplan1958estimation | [Original paper](https://web.stanford.edu/~lutian/coursepdf/KMpaper.pdf) | Nonparametric survival estimation under censoring. |
| AJ / therneau2026competing | [Official survival vignette](https://cran.r-project.org/web/packages/survival/vignettes/compete.pdf) | AJ state probabilities, CIF and relationship to KM. Used checked methodological documentation rather than pretending an inaccessible 1978 paper was read. |
| Competing-risk AUC / blanche2013competingauc | [Original publisher abstract and metadata](https://onlinelibrary.wiley.com/doi/10.1002/sim.5958) | IPCW AUC and alternative competing-event control definitions; manuscript names the chosen one. Does not verify our software. |
| Censored Brier / gerds2006brier | [Original publisher abstract and metadata](https://onlinelibrary.wiley.com/doi/10.1002/bimj.200610301) | Censoring-weighted prediction error; assumptions on the censoring mechanism matter. Not a blanket validation of every competing-risk implementation. |
| CTCAE / nci2025ctcae | [NCI official v6.0 document](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf) | Clinical grading reference; source version and deviations must be documented per endpoint. No implication that all registry/laboratory outcomes are validated CTCAE adverse events. |

The Introduction's existing primary sources cover the clinical score examples,
prior haematology models, rarity, shared learning, meta-learning and pretrained
EHR representations. Descriptions of what OPERA tests and conditional potential
uses are the study rationale, not literature findings requiring an unrelated
citation. No additional performance claims were added in this pass.

Claims still requiring our evidence, not citations: exact input modalities,
clinical-code availability at index, all cutoffs/eligibility, governance details,
completed fitting, selection locks, nation-scale event cutoff and training
population, and seed-summary/primary estimand definitions. Literature cannot
turn these pending fields into completed methods. This is a focused claim audit,
not a systematic review or verification of every bibliography entry.

CORE-BEHRT attribution: the user confirmed it is the modelling foundation.
Introduction and encoder Methods now state this directly and distinguish the
causal next-event/multi-outcome adaptations from the original BERT-based work.
Bibliography metadata was checked against the original PMLR publication
https://proceedings.mlr.press/v252/odgaard24a.html and updated from the arXiv
record to MLHC 2024, PMLR 252, including the proceedings author order.

The expanded plain-language explanation of the IPI/NCCN-IPI variable families
was checked against the original NCCN-IPI article and publication record:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5527396/ and
https://pubmed.ncbi.nlm.nih.gov/24264230/. The manuscript describes variable
families, not scoring cutoffs. It does not transfer this score definition to
CLL-IPI or myeloma staging. The reason for choosing haematology is stated as
the study rationale, not a claim of experimentally established shared biology.
