# Manuscript review plan, 18 October 2026

Goal: before results arrive, make sure the paper is logically tight, every
claim is supported by a prespecified analysis, and the strongest likely
critiques are answered in the design or the text.

## Pass 1: logic of the argument
1. **Questions and answers.** Every Results subsection asks a question that
   the Introduction motivates, and each is answered by a prespecified analysis
   named in Methods.
2. **Claims and support.** Every claim in the abstract, key findings and
   Discussion has a number slot (`\result`) or a citation, and no claim
   depends on a result we do not compute.
3. **Methods and analysis code.** Every analysis described in Methods exists in
   the analysis code, and every reported table or figure has a source.
4. **Numbers.** Counts agree across the abstract, Methods, Results, Table 1 and
   the flow diagram (37,531; 29,105; 1,980; 6,446; 24 subtypes; 69 outcomes;
   52 endpoints; eight subtypes without an index).
5. **Terminology.** Held-out test set, tuning set, joint, routed and
   specialist, cell, endpoint and domain are each used in one sense.

## Pass 2: design weaknesses a reviewer will find
6. Data scientist: effect size and meaning, tuning-set size, single fits, the
   source of the gain, leakage through pretraining, the confounded national
   comparison, multiplicity.
7. Clinician: prediction time and intended use, index endpoint, outcome
   proxies, temporal shift in treatment, rare subtypes, clinical usefulness,
   what the model reads.
8. Generalisation: Denmark only, register latency, no external validation.

## Pass 3: figures and tables
9. Each figure answers one question, and its caption says how to read it.
10. Supplementary figures are cited in order and each earns its place.
11. Table 2 and the supplementary tables can be generated from the outputs.

## Rule for changes
- Clear improvements (consistency, wording, a missing prespecified stratum,
  a missing citation): implement and report.
- Consequential choices (new endpoints, new primary analyses, dropping or
  promoting figures, changes to the analysis lock): flag for decision.
