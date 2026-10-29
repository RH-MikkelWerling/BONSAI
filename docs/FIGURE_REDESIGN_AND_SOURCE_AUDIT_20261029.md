# Figure redesign and source audit — 29 October 2026

This follows the graphical rehearsal and the Danish-code audit of 28 October.
It uses the downloaded tuning bundle, repository R scripts, and the two public
producer repositories. Frozen outcomes, checkpoints and held-out data have not
been changed. User confirms retaining the existing script definitions and
reporting ascertainment limitations in the supplement, without rebuilding
outcomes or refitting models (29 October). Server provenance still needs to establish the exact installed
versions. A current public repository is evidence about its implementation,
not proof of the version used by an older frozen build.

## What the producer repositories resolve

The inspected revisions are ehr2meds
`1594b8d9c4921ecd97ba63b868c30ac3f1203cda` and DALY-CARE
`b9c12e7261df868d7e5c9653841b95dea9b05862`. The fetched-file inventory, pinned
URLs and blob identifiers are in
`scratch/external_code_audit_20261029/source_manifest.json`.

- The [lymphoma pipeline](https://github.com/FGA-DIKU/ehr2meds/blob/1594b8d9c4921ecd97ba63b868c30ac3f1203cda/configs/MEDS/lymphoma_pipeline.yaml)
  fits adaptive counts and numeric transforms on the 90% conversion-training
  patients' events before 2022. These patient partitions serve conversion
  statistics and are distinct from the clinical temporal split. The default
  adaptive minimum is 100 events. User confirms this configuration family and
  adaptive truncation were used. Its exact frozen build is still to be checked.
- [Adaptive hierarchy lengths](https://github.com/FGA-DIKU/ehr2meds/blob/1594b8d9c4921ecd97ba63b868c30ac3f1203cda/configs/MEDS/default_adaptive_code_mapping.yaml)
  include pathology prefixes of five, four and three characters. Rare
  `FE16E3` and `FE16E0` can both map to `FE16E`, losing positive/negative-result
  specificity. Common full codes are retained. A rare code without a qualifying
  ancestor is also retained. We must inspect the mapping rather than assume
  every rare code was truncated.
- [Numeric binning](https://github.com/FGA-DIKU/ehr2meds/blob/1594b8d9c4921ecd97ba63b868c30ac3f1203cda/ehr2meds/meds_stages/annotate_numeric_values.py)
  clips to fitted percentiles, normalises, and assigns zero-based, right-sided
  quantile bins. These are not clinical normal/abnormal categories. An edge at
  zero can make bin zero unreachable after clipping. The audit now handles
  this case and constant assays explicitly.
- [DALY-CARE NPU constants](https://github.com/RH-CLL-LAB/dalycare_package_v2/blob/b9c12e7261df868d7e5c9653841b95dea9b05862/NPU_codes.R)
  resolve the external analyte vectors. Phosphate `NPU.FOS` is NPU03096.
  Magnesium is named `NPU.MG2` in the package, while OPERA defines its local
  `NPU.MAG`. The earlier claim that these constants could not be inspected is
  superseded.
- [Laboratory cleaning](https://github.com/RH-CLL-LAB/dalycare_package_v2/blob/b9c12e7261df868d7e5c9653841b95dea9b05862/functions/clean_lab_values.R)
  converts albumin NPU01132 by ×0.0665, CRP NPU01423 by ×0.105, and urate
  NPU09356 by ÷1000 under its unit conditions. OPERA's R endpoint script calls
  this function and uses its cleaned value. The MEDS numeric conversion does
  not automatically call the R cleaner, so outcome units cannot simply be
  assumed for encoded lab tokens.
- [Molecular loading](https://github.com/RH-CLL-LAB/dalycare_package_v2/blob/b9c12e7261df868d7e5c9653841b95dea9b05862/load_snomed_molecular.R)
  identifies historical del17p `M632C0` and MYC translocation `M636D0`. These
  exact aliases were added to the provisional attribution configuration and
  preserved during roll-up. `M632C0` also has a public Patobank del17p13 label.
  `M636D0` is package-defined evidence, not independently verified here in the
  current public Patobank pages. Generic t(8;14) `J08141` remains excluded from
  the MYC-positive concept pending assay-level confirmation.

## Which code concerns affect the actual 69 outputs?

The current output roster is visible in the saved 69 outcome-source names and
atlas. It must not be inferred from the larger historical experiment registry.
The release policy is `opera/configs/outcome_release_primary_v2.yaml`, with
adjudications in `outcome_panel_decision_v1.yaml`.

| Concern | In current outputs? | Finding and remaining check |
|---|---|---|
| “B89” | No such reference found in the operative code searches | The earlier issue concerns **B59**, not B89. |
| Pneumocystosis B59 without B48.5 | Component of serious infection and its broad union | R uses B59 only. SKS closed B59 in 2018 and introduced B48.5 in 2019. Need frozen-script lock and aggregate counts of otherwise missed eligible B48.5-only cases. Added a conditional ascertainment note to the supplement. |
| Broad B44 fungal selection | Component of serious infection | Includes colonisation and other non-invasive forms. Standalone invasive-fungal outcome is absent from the current 69, but its code selection still matters inside the composite. Use “selected fungal diagnoses” in the definition. |
| Full I80 | VTE remains active | Includes superficial thrombophlebitis. User confirms retaining the code set. The supplement now states its breadth. |
| BOQA red-cell label | Red-cell and any-transfusion outputs active | Public procedure codes establish blood transfusion, not product specificity. User confirms an SKS-only basis and is unsure about product specificity. The supplement explicitly qualifies it as a BOQA-based proxy. |
| MACE composite | **Excluded** | Explicitly excluded by the primary-v2 policy and absent from the exported roster. The broad historical definition is not an active primary result. |
| Full G45 stroke/TIA | Active | Broader transient syndromes remain part of the definition. |
| D47 in subsequent haematological malignancy | Active | Includes uncertain/precursor conditions. User confirms retaining this code set. The supplement explains uncertain/precursor conditions and that a new block is an operational event. |
| Mixed fibrinogen units | **Retired** | Fibrinogen grade outcomes are retired for sparse support and unresolved assay/unit grading. Do not present this as a defect in an active output. |
| Mixed-fluid LDH | Elevated LDH active | NPU19658 is plasma, NPU19978 pleural fluid, NPU19975 ascitic fluid. User thinks inclusion may be deliberate but is unsure (29 October reply). Keep the frozen vector, record clinical interpretation as unresolved, and keep these assays distinct in attribution labels. |

The official identity evidence remains in
[the Danish-code audit](DANISH_CODE_AUDIT_20261028.md). The open-question CSV has
been updated to distinguish repository-resolved questions from checks that
still need frozen-build or clinical evidence.

## Why the borrowing and disease-bank plots looked wrong

Three actual plotting errors were fixed:

1. The raw-vs-shrunken diagnostic replaced the shrunken values with raw values
   before plotting, creating a misleading identity relationship. It now keeps
   the original posterior estimates for that diagnostic.
2. The disease-bank heatmap reused target labels for both axes. **NMZL is a
   source but has no target row in this export.** The plot dropped that source
   and renormalised the remainder. It now uses all 24 source columns and the
   23 observed target rows, and hatches home cells by label identity rather
   than by diagonal position. Regression coverage checks the preserved shares.
3. The source-removal maps had the same target-only axis assumption. Their
   tables contain 20 estimable targets and 24 sources. The maps now retain
   amyloidosis, NMZL, SMZL and solitary plasmacytoma as source columns even
   without target rows. Domain matrices also retain source-only lineage groups.
   These display fixes use the existing removal results without recomputation.

The saved source-removal tables themselves support a separate, substantive
finding. For the mean-ablation all-outcome pairs, 452 finite off-diagonal
estimates have median absolute raw effect about 0.0603 AUC points and maximum
about 3.111 points. The adaptive prior collapses close to zero. Its maximum
absolute posterior mean is about 0.0000651 points. Reapplying the saved
estimator reproduces these values to numerical precision. This is not a
factor-of-100 display error. Showing a tight colour range would exaggerate
the posterior effects.

The review recipe now shows unshrunk pairwise effects with an identical ±2
AUC-point scale for mean and zero replacements, no pairwise significance dots, and an explicit shrinkage
diagnostic. Mean and zero ablations remain separate, labelled sensitivities.
Saturated heatmaps have colourbar extension triangles. Source-alone effects
are not additive AUC contributions and must not be read as evidence of a
causal relationship between diseases.

The exported gate weights pass range, normalisation and duplicate checks.
They span 24 disease sources, 69 outcome sources and one shared slot. The old
plot-table export discarded model/seed/split fields, so these aggregates alone
cannot prove the source ordering inside the checkpoint. Future tables retain
those fields. The source-removal provenance names R_AB_L and scientific hash
`79d39ad1e7cb514720265e271266a3385a713e7c6fbd7f730974da053c059e95`.
Mean and zero tables differ. A provenance list containing both branches is
expected from its recursive input inventory and is not evidence that the
renderer selected the wrong table.

Gate allocation remains broad and outcome-heavy. The revised bank plot shows
the actual assigned mass and labels entropy of **mean** gates separately from
patient-level entropy. Stable mean gates do not establish selective borrowing
or predictive benefit. Verifying the checkpoint's bank ordering requires the
server-side model/configuration and source inventory, not cosmetic changes.

## Figure changes for review

- Restored the previous headline reference ladder, patient-map contours and
  rescue scatter/forest arrangement. A separate all-model overview preserves
  comparison of available OPERA variants without crowding the headline.
- Added one-disease-at-a-time patient-map facets on identical coordinates.
  The main panel retains the familiar contours. Its legend gains space with
  the number of diseases, rather than shrinking the text.
- Clinical comparison legends now identify model-derived versus index groups,
  use larger text and distinguish four native groups with line style as well
  as colour. Precision–recall summaries and group keys have reserved space.
- Calibration slopes use the complete finite IQR range. Very long tails use
  a labelled symmetric-log axis, with no truncation of intervals. Calibration
  maps use a shared range covering supported residuals by default. Explicit
  bounded ranges are marked if saturated.
- Router bank C shows other-disease, other-outcome and own/shared allocations.
  D is labelled as breadth of the mean profile. Legends and explanatory notes
  occupy separate areas.
- Strategy bars accept specialist, disease-pooled, outcome-pooled, joint and
  OPERA. Selection is by tuning log loss. The expanded bar policy is separate
  from the original S/D/O comparator in B. Unknown selected models now fail
  clearly instead of disappearing from the bars.
- Recent-record omission plots use the shared typography, endpoint labels,
  colours and interval treatment. Panel-title size defaults to the common
  style size rather than a competing hard-coded size.

Revised PDFs/PNGs and the graphical review page are in
`scratch/paper_reading_20261028/graphical_audit_20261028/redesign_20261029/`.
The [updated gallery](../scratch/paper_reading_20261028/graphical_audit_20261028/figure_review_20261029.html) contains 80 suite renders and five extra stress probes. Each image states whether it uses tuning aggregates or synthetic layout data.
Clinical patient-level inputs, full embeddings and full attribution inputs
were not in the downloaded bundle. Their local previews validate layout, not
new clinical results. Server regeneration is still required for those panels.

## What is still needed from the server

Use the saved producer artifacts, not newly fitted replacements:

1. Frozen outcome-build lock and source/package revision. Quantify eligible
   B48.5-only cases and overlap with the existing composite if readily available. This is an
   optional aggregate ascertainment check, not a request to regenerate outcomes.
   VTE, transfusion-product and D47 limitations are now documented.
2. `adaptive_code_mapping.parquet` and `numeric_metadata.json` from the fitted
   lymphoma build. These answer retained molecular specificity and physical
   quantile boundaries. They do not require rebuilding data or retraining.
3. Re-render figures with the corrected code using the finished server analyses and
   updated clinical inputs. Keep model/config/source inventories with the
   outputs so that each plot identifies its actual fit.

The vocabulary audit now accepts both artifacts:

```sh
python -m opera.run.audit_vocabulary \
  --vocabulary "$OPERA_VOCABULARY" \
  --config opera/configs/attribution_analysis.example.yaml \
  --adaptive-mapping "$OPERA_MEDS_BUILD_DIR/adaptive_code_mapping.parquet" \
  --numeric-metadata "$OPERA_MEDS_BUILD_DIR/numeric_metadata.json" \
  --output-dir "$OPERA_ANALYSIS_DIR/vocabulary_audit"
```

Set these paths to the frozen build, vocabulary and analysis directory.
`adaptive_concept_coverage.csv` lists lost/retained specificity and competing
original codes. `laboratory_bin_definitions.csv` lists boundaries in input
units, reachable bins and vocabulary presence. They contain no patient rows,
but fitted numeric boundaries are data-derived summaries and need the normal
export review. Provisional concept approval remains **false**.

## Verification record

The 34 production recipe signatures validate. The 80-figure rendering suite
completed, including ten clinical-index families, four-category native scales,
a sparse/tied-risk case, and fallback typography. Five additional probes cover
10/24 disease labels, matching disease facets and extreme calibration slopes.
The entire inventory was reviewed as contact sheets, with full-size inspection
of the changed borrowing, router, clinical, strategy, patient-map and
calibration layouts. Automated collision detection is a screening tool, not
a guarantee for future data. The 24-disease contour stress case illustrates
why identification still needs the faceted companion.

The manuscript rehearsal and standalone outcome-definition supplement compile.
These are syntax checks in the dated scratch folder, not a replacement of the
main reading PDF with final server figures.

Final regression pass: **124 tests passed** across figure identity/layout,
code/concept mapping, attribution summaries, paired analyses, plot-table
building, rendering, source removal, manuscript numbers and clinical metrics.
All 85 PDF hashes match their render records. The layout screen reports zero
text-collision candidates and zero legend overflows for these rendered inputs.
The code manifest has been refreshed and `git diff --check` passes.
