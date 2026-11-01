# Positioning the new OPERA design (literature check, 1 November 2026)

Web search, citations checked against publisher, ACL Anthology, PMLR, arXiv or
institutional pages (not a systematic database search; 2025–26 hematology ML
literature not exhaustively covered). Details from abstracts only are flagged.

## Closest prior work

**Label-wise attention.** Mullenbach et al., Explainable prediction of medical
codes from clinical text (CAML), NAACL 2018, doi:10.18653/v1/N18-1100. Vu et
al., A label attention model for ICD coding (LAAT), IJCAI 2020. *Delta:* OPERA
applies a learned query per outcome to structured event sequences, as a gated
residual on a shared summary, for competing-risk outcomes. No peer-reviewed
per-outcome query pooling for multi-outcome survival on structured EHR found
(Shickel et al., arXiv:2111.05431, may use task tokens; unconfirmed).

**Many-outcome EHR models.** Harutyunyan et al., Sci Data 2019. McDermott et
al., A comprehensive EHR timeseries pre-training benchmark, ACM CHIL 2021,
doi:10.1145/3450439.3451877 (multitask fine-tuning often shows negative
transfer). Steinberg et al., MOTOR, ICLR 2024 (time-to-event pretraining over
thousands of codes). Wornow et al., EHRSHOT, NeurIPS 2023. Shmatko et al.,
Delphi-2M, Nature 2025, doi:10.1038/s41586-025-09529-3 (Danish validation).
Gadd et al., SurvivEHR, npj Digit Med 2026, doi:10.1038/s41746-026-02709-z
(competing-risk CIF foundation model; journal page not opened). Chandak et al.,
EveryQuery, ICML 2026, arXiv:2603.07900 (query-conditioned zero-shot
prediction; nearest analogue of query-conditioned reading — OPERA uses a fixed
learned query per outcome plus task corrections). Background: BEHRT (Sci Rep
2020), Med-BERT (npj Digit Med 2021), Foresight (Lancet Digit Health 2024),
TransformEHR (Nat Commun 2023).

**Two-axis tasks.** Yang & Hospedales, A unified perspective on multi-domain
and multi-task learning, ICLR 2015 (domain × task descriptors; precedent for
diagnosis × outcome and the factorized correction); Deep multi-task
representation learning: a tensor factorisation approach, ICLR 2017. Sheng et
al., STAR, CIKM 2021 (shared network + per-domain parameters).

**Shared + task-specific components and gating.** MMoE (Ma et al., KDD 2018);
PLE (Tang et al., RecSys 2020; shared vs task-specific experts, "seesaw");
adapters (Houlsby et al., ICML 2019); AdapterFusion (Pfeiffer et al., EACL
2021). *Delta:* our null result for learned routing is an empirical
counterpoint for clinical data, consistent with McDermott et al.

**Partial pooling across clinical subgroups.** Suresh et al., Learning tasks
for multitask learning: heterogenous patient populations in the ICU, KDD 2018.
Wiens et al., JAMIA 2014 (transfer across hospitals). Zhang et al., MetaPred,
KDD 2019. No neural clinical model with explicit n/(n+τ) shrinkage on task
adapters found.

**Lymphoma.** Biccler et al., JCO CCI 2018 (DLBCL ML vs IPI); Rask Kragh
Jørgensen et al., JCO CCI 2024 (Hodgkin); Agius et al., CLL-TIM, Nat Commun
2020 (Danish CLL infections). All one subtype per model; no pan-lymphoid
many-outcome model against subtype-specific indices found.

## Claim wording (proposed)

"To our knowledge, this is the first model to jointly predict competing-risk
outcomes across a full diagnosis-by-outcome grid of lymphoid cancers, combining
a shared pretrained encoder, outcome-specific attention pooling and additive
diagnosis- and outcome-specific corrections; we further show that learned
routing between tasks gives no benefit over this simpler design."

Do not claim novelty for label attention, adapters or factorization themselves.
