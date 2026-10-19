# Submission readiness, 18 October 2026

What stands between the current manuscript and submission, in order.

## A. Fills itself from the locked results (no writing needed)
- 71 `\result` keys in the main text, and Table 2, come from
  `paper-numbers`. Figures come from `figures` and `clinical-scores`, and the
  printed supplementary tables from `supplement-tables` (RUNBOOK, "From
  locked results to a filled manuscript").
- Check after filling: a search for "[" in the PDF lists every slot still
  open.

## B. Needs a decision or a sentence once results are known
- The direction words in the Results ("complementary / redundant", "within
  / outside", "did / did not"). About 25 slots, each next to its number.
- Discussion: principal findings, interpretation, and the conclusion (three
  placeholders).
- Abstract: the last two sentences (national encoder, where sharing failed)
  and the conclusion.
- Key findings box: confirm each bullet once its number is in. The bullet on
  data scarcity is conditional.

## C. Needs author input now (independent of results)
1. Affiliations of the authors marked with an asterisk.
2. PHAIR encoder: number of individuals and events, and a reference.
3. Local encoder: individuals, events, training time and selected epoch.
4. Practical margin (0.01) in the analysis lock.
5. Table S1: coverage years for each register.
6. First-line treatment changes in Denmark and when (Discussion, Results).
7. Register coverage changes over time (laboratory data by region and year).
8. Done: the ten subtypes without a dedicated index are registry identifiers in
   the locked analysis config, tested against the taxonomy. Still to do: check
   the guideline column of the index table against the Danish guidelines.
9. Feature export check for the tabular models (Supplementary Methods).
10. The selected router configuration; locked priors, tolerances and
    support-bin edges.
11. Index inventory (versions, populations, availability) from the secure
    analysis.
12. Repository URL, archival DOI and licence for the code; licence of
    ehr2meds.
13. Funding, acknowledgements, CRediT contributions and competing interests.
14. **Code names and code lists (open, decide how).** A way to give the
    analysis the meaning of the model's codes without patient data leaving the
    server: ICD-10, ATC, NPU, SKS and PATOBANK SNOMED name tables (public
    classifications), and the codes of the molecular pathology findings (TP53,
    MYC, IGHV) in both the cohort and the national vocabulary. Options: export
    the vocabulary audit's `code_meanings.csv`/`search_hits.csv` (codes and
    names only, no counts) after disclosure review; or copy the official
    tables into the secure environment and list them under `code_tables:`.

## D. Journal requirements to check (HemaSphere and top-tier alternatives)
- Display items: currently 7 main figures and 2 tables. The HemaSphere DLBCL
  paper had 10; confirm the limit in the author guidelines.
- Word count of main text and abstract. The abstract is about 290 words with
  placeholders; trim to the journal limit (often 250).
- A completed TRIPOD+AI checklist as a supplementary file (the crosswalk in
  `paper/TRIPOD_REPORTING_CROSSWALK.md` is the starting point).
- Ethics: approvals are stated in Methods. Add a consent statement (register
  research without individual consent under Danish law) if required.
- Data and code availability: present. Code release with the synthetic demo
  (recommended) before submission.
- Patient and public involvement statement (required by BMJ- and
  Lancet-family journals).
- Cover letter: why now, what is new (the learning-design decomposition at
  national scale, indices on their own endpoints, a held-out test set across
  a treatment shift), and a suggested reviewer list mixing clinical machine
  learning and lymphoma.

## E. Pre-submission build
1. Copy generated numbers, figures and tables into `paper/` (RUNBOOK).
2. `latexmk -pdf opera_manuscript.tex` (no synthetic preview notice).
3. Search the PDF for "[" and "SYNTHETIC"; both should be absent.
4. Disclosure review of every exported number (counts below 5 suppressed;
   the numbers JSON lists every value with its source).
5. Supervisor and co-author read; then journal formatting.
