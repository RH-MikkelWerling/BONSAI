# Supplement review — 28 October 2026

Reviewed the active model-details, evaluation-details, supplementary figure captions and standalone data-definitions supplement. The superseded model-reporting and ascertainment checklists are historical documents, not current reader-facing methods.

## Corrected

- Removed the assertion that the final test evaluation had already been completed once under a locked plan. Current previews are tuning analyses. Final lock and amendment manifests remain required.
- Separated equal-cell and disease-balanced evaluation weights from the 52 endpoint families used for training loss.
- Made NCCN-IPI the main DLBCL comparator. Clinical groups use training-derived thresholds, frozen for subsequent evaluation. Separate panel-wide operating points still use tuning data.
- Described native categories, unsupported boundaries, apparent rather than out-of-fold training predictions, and classification metrics at every non-low-risk boundary. The new clinical CSVs contain point estimates, not confidence intervals.
- Removed the unused decision-curve figure and its prespecification claims.
- Clarified that disease-source ablations retain the outcome bank, global correction and shared encoder. They are conditional analyses of a fitted model, not retraining experiments or biological effects.
- Removed guarantees of calibration from natural sampling, architectural equivalence from a two-fit comparison, and absence of selection bias from choosing a lower tuning score.
- Corrected the current router caption to both adapter banks rather than the earlier disease-only selection.
- Distinguished prediction-coloured patient galleries from optional observed-risk kernel maps. Added recent-record gap analyses to input restriction.
- Removed claims of completed explanations for every held-out patient and universally reviewed concept labels.
- Clarified outcome-panel arithmetic: 60 unchanged plus 7 revised retained outcomes, then 2 additions, gives 69.
- Replaced the dated definitions-PDF include with `render/definitions_current/` and rebuilt from the current definitions source.
- Replaced OPERA variant subscripts with parenthetical local, national and national + adaptation labels. Internal experiment identifiers and contrasts are unchanged.

## Must resolve before submission

1. **Preprocessing chronology.** An encoder's pre-2022 event cutoff does not establish that vocabulary frequencies, rare-code filtering or laboratory bins were fitted before that date. National vocabulary statistics may cover a wider period. Obtain the actual preprocessing manifests for both encoders and report any temporal contamination explicitly.
2. **Exact checkpoint/input contract.** Confirm separator-token policy, actual completed pretraining epochs, parameter counts, optimiser settings and the selected checkpoint hashes. Historical configuration descriptions do not establish the inputs of a trained checkpoint.
3. **Clinical provenance and completeness.** Confirm native score versions/categories and recorded versus computed score provenance for every index. Already-coarsened registry categories cannot be reconstructed from labels alone. Report missingness by period and the comparison population, including unsupported indices.
4. **Clinical threshold uncertainty.** Training predictions are apparent predictions. Larger training groups stabilise estimates but do not remove optimism or uncertainty in the cutoffs. Current threshold-metric exports lack confidence intervals. Check small-count disclosure before exporting those CSVs; they do not yet enforce suppression automatically.
5. **Outcome implementation.** Verify the installed DALY-CARE grading version, urate units and calcium correction. The CTCAE table records mixed version-specific adaptations and unresolved items, not certification of exact CTCAE equivalence. Check retained/new/renamed identifiers against the exported 69-outcome manifest.
6. **Post-index subtype assignment.** The 209-patient sensitivity addresses later assignments but does not establish that every other disease assignment was known at prediction time. Audit early post-index assignments and disease-conditioned router inputs.
7. **Final analysis status.** Check the full prespecification/amendment history, source-removal manifests, primary router identity, seed controls and final held-out evaluation before changing planned analyses to completed findings. Do not infer national performance from local results.
8. **Inference details.** Verify finite-bootstrap support rules, weighting for each hierarchy and shrinkage map, censoring-weight handling and hierarchical priors/diagnostics from the executed configuration. Patient-bootstrap intervals condition on fitted models and do not include general training-run uncertainty.
9. **Attribution provenance.** Record actual processed patients, split, outcomes, quadrature/refinement settings and completeness failures. Verify concept-review status, expectation counts and sanity-check completion. Recent-week removal does not replace integrated-gradient faithfulness checks.
10. **Figure/caption alignment.** The paper still contains staged descriptions of final held-out figures and placeholders. Inspect each refreshed server artifact against its caption: endpoint, horizon, sample, summary statistic, prediction versus observed colour and arrow projection. Rebuilding TeX does not refresh server figures.

## Rebuild

From `paper/`, build definitions first, then the reading manuscript:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=render/definitions_current opera_data_definitions_supplement_embedded.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../scratch/paper_reading_20261028 opera_manuscript_reading.tex
```

The historical `definitions_20261014` PDF is no longer the active include. No server rerun or new empirical result was produced by this editorial audit.
