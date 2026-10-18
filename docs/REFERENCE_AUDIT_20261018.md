# Reference and claim audit — 18 October 2026

Scope: main manuscript, both inline methods supplements and the appended data-
definition/model-reporting sources. All citation keys are checked structurally
against `paper.bib` and in the compiled documents. The table below records the
claim-bearing sources revisited in this pass. A working citation key or DOI is
not a certification that every possible claim about a paper is supported.
Publisher access failures are not treated as successful full-text verification.

## Substantive findings and corrections

| Source/claim | Finding and manuscript disposition |
|---|---|
| [MZL-IPI, Arcaini et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019091/) | The paper reports 501 eligible development patients and external validation in a pooled US cohort of 353. Removed its use in a blanket claim that rare-disease scores have only tiny cohorts/little external validation. Published existence and routine adoption are separate questions. |
| [PCL, Jurczyszyn et al.](https://onlinelibrary.wiley.com/doi/10.1111/bjh.15092) | The source supports a 117-patient development cohort and calls for independent validation. Retained as a specific example, not evidence that all rare-subtype scores are unvalidated. |
| [Richter, Tsimberidou et al.](https://ascopubs.org/doi/abs/10.1200/JCO.2005.05.0187) | The original study describes 148 confirmed cases and prognostic factors; appropriate as a small rare-disease study, not as proof of current routine adoption. |
| [Danish DLBCL guideline](https://www.dmcg.dk/kliniske-retningslinjer/kliniske-retningslinjer-opdelt-paa-dmcg/lymfom/diffust-storcellet-b-celle-lymfom/) | Supports clinical risk stratification and CNS-IPI work-up. The guideline's use of scores does not validate substituting a newly developed treatment-start predictor for its treatment rules. |
| [Earlier DLBCL study](https://onlinelibrary.wiley.com/doi/10.1002/hem3.70336) | Verified title, authors, endpoint framing and the reported 160 versus one older low-risk patients. That is an earlier cohort/definition, not a result of OPERA. Its feature pipeline motivates the comparator; the final server feature export still needs matching. |
| [NCCN-IPI original](https://mdanderson.elsevierpure.com/en/publications/an-enhanced-international-prognostic-index-nccn-ipi-for-patients-/) | Publication identity/rituximab-era population checked; retained original citation instead of calling it universally “more robust” based on one local comparison. Fixed local LDH limits are a study implementation choice requiring codebook review. |
| [FLIPI original](https://pubmed.ncbi.nlm.nih.gov/15126323/) | Added missing original source: Blood 104, 1258–1265 (2004), DOI 10.1182/blood-2003-12-4434. Development and confirmation cohorts are described by the source. |
| [MIPI original](https://ashpublications.org/blood/article/111/2/558/103675/A-new-prognostic-index-MIPI-for-patients-with) | Added original source, Blood 111, 558–565 (2008). Publisher-indexed text supports advanced-stage MCL and OS; direct page retrieval was blocked. |
| [IPS original](https://www.imise.uni-leipzig.de/Static/Archiv/1998/hasenclever-d-1998-1506-a.pdf) | Added Hasenclever/Diehl original source. Freedom from progression is not identical to this study's second-line-treatment/death proxy. The comparison is now explicitly endpoint-approximate. |
| [ISS original](https://ascopubs.org/doi/abs/10.1200/JCO.2005.04.242?role=tab) | Added Greipp et al. original reference. An ISS registry label must not be silently treated as R-ISS; preserve exact version in the clinical inventory. |
| [CTCAE v5](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v5-5x7.pdf), [NCI current resources](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events) | Added v5 original because upstream grading is partly v5-derived. NCI confirms v6 release in 2025. Neither version validates locally reconstructed laboratory-only grades as adjudicated clinical toxicity. |
| [CORE-BEHRT](https://proceedings.mlr.press/v252/odgaard24a.html) | Official PMLR record confirms authors, volume 252 and architecture/data/training optimisation focus. Preserve credit; OPERA's causal next-event adaptation is described separately from the original bidirectional model. |
| [BONSAI](https://joss.theoj.org/papers/10.21105/joss.08869) | Added the software citation from the JOSS record: 10(114), 8869 (2025), with original author list. |
| [Numeric encoding](https://proceedings.mlr.press/v306/montgomery26a.html) | Updated arXiv-only entry to published ICML/PMLR 306, 90118–90133 (2026). The paper supports comparison of numeric encoding strategies and an empirical bin-count rule; the exact implemented constants/statistic remain implementation provenance rather than established clinical truth. |
| [MetaPred](https://arxiv.org/abs/1905.03218), [Time Associated Meta Learning](https://arxiv.org/abs/2303.02570) | Titles/authors and clinical transfer framing checked. These motivate sharing/adaptation, not a claim that OPERA implements meta-learning. |
| [EveryQuery v3](https://arxiv.org/abs/2603.07900v3) | Current v3 title/authors/revision date match the existing bibliography. It estimates query-conditioned predictions, not evidence that disease/outcome pooling necessarily benefits this cohort. |
| [Harutyunyan et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6572845/), [McDermott et al.](https://arxiv.org/abs/2007.10185), [Ding et al.](https://psb.stanford.edu/psb-online/proceedings/psb19/ding.pdf) | Revisited the heterogeneous-benefit argument. Ding directly supports rare-phenotype gains and common-phenotype harm; its phenotyping setting is distinguished from future clinical-event prediction. |
| [Delphi-2M](https://www.nature.com/articles/s41586-025-09529-3) | Original source supports UK Biobank learning and evaluation in Danish register data. Avoid claims that cross-system evaluation is intrinsically unhelpful. |
| [MSK-CHORD](https://www.nature.com/articles/s41586-024-08167-5) | Original source supports multimodal real-world integration and cancer outcome prediction; it is not a transformer disease/outcome pooling ablation. |
| [MOTOR](https://proceedings.iclr.cc/paper_files/paper/2024/file/b36554b97da741b1c48c9de05c73993e-Paper-Conference.pdf), [EHRSHOT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d42db1f74df54cb992b3956eb7f15a6f-Abstract-Datasets_and_Benchmarks.html) | Publication records and aims checked; added accessible proceedings links. MOTOR is ICLR 2024 despite the historical `2023` citation key. EHRSHOT provides few-shot evaluation, not direct evidence for this clinical panel. |
| [SurvivEHR](https://www.nature.com/articles/s41746-026-02709-z) | Added a closely relevant May 2026 paper on competing-risk next-event pretraining in primary-care records. This strengthens the need to frame OPERA's contribution as the controlled clinical comparison, not first use of broad competing-risk EHR prediction. |
| [Houlsby adapters](https://proceedings.mlr.press/v97/houlsby19a.html), [AdapterFusion](https://aclanthology.org/2021.eacl-main.39/), [LoRA](https://arxiv.org/abs/2106.09685) | Original sources motivate parameter-efficient modules and composition. Low-rank residual bottlenecks are not automatically the exact LoRA algorithm; the manuscript describes its own implementation. |
| [Integrated gradients](https://proceedings.mlr.press/v70/sundararajan17a.html), [sanity checks](https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) | Added both original references. Completeness, reference dependence and randomisation motivation do not establish causality, biological correctness or clinical trustworthiness. Partial parameter randomisation is labelled accurately. |
| [IPA](https://link.springer.com/article/10.1186/s41512-018-0029-2), [competing-risk accuracy](https://pubmed.ncbi.nlm.nih.gov/21259311/), [validation sample size](https://onlinelibrary.wiley.com/doi/10.1002/sim.6787) | Source identities and methodological purpose checked. A rough event-count rule is not a guarantee of validation precision; hierarchical panel benefit does not certify sparse individual cells. |
| [CLL-TIM deployment](https://www.nature.com/articles/s41746-024-01132-6) | Supports EHR deployment and an MDR assessment. Replaced an unconditional regulatory assertion with the need for appropriate assessment; retrospective register compatibility is not deployment readiness. |
| [timeseriesflattener](https://joss.theoj.org/papers/10.21105/joss.05197.pdf), [TRIPOD+AI](https://www.bmj.com/content/bmj/385/bmj-2023-078378.full.pdf) | Original records checked. A reporting guideline is a framework for reporting, not evidence that every item has already been fulfilled. |

## Items that citations cannot resolve

- Exact current “no routinely used score” subtype classification and Danish
  adoption need clinical review, not inference from absence in a registry.
- National encoder population, release, tokenisation/mapping, training budget and
  dates require its source artifacts and owner-confirmed provenance.
- Data cutoff, sample counts, treatment-line accuracy, grading implementation and
  diagnosis availability require data/implementation review.
- Local LDH normal limits, score dates and contemporaneous clinical-score versions
  need the clinical metadata/codebook; citing an original score cannot establish
  that the local reconstruction followed it.
- Historical dates of Danish treatment adoption need Danish records/guidance;
  POLARIX alone does not date local implementation.

The audit does not claim line-by-line independent replication of every cited
paper. Existing registry/theoretical background references not revisited in the
table retain their bibliographic entries; an uncaught claim–source mismatch must
not be treated as resolved merely because LaTeX builds. The most consequential
claims, new references and previously overextended citations are documented above.
