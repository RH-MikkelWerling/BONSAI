# Clinical endpoint groups for descriptive plots (v1 proposal)

`opera/configs/outcome_display_groups_v1.yaml` defines 17 display-only groups
over the 87 outcomes. Every outcome occurs exactly once; the largest group
contains eight outcomes. This replaces the broad 25-outcome electrolyte and
metabolic family **in figures only**. It does not change the 11 families in the
registry, the grade-pairing `endpoint_groups_v1` used by the current low-rank
BCE loss, checkpoint selection, or any existing model score.

The main clinical splits are infection proxies versus serious/opportunistic
infection; erythroid/transfusion, neutrophil/WBC, lymphocyte/leukocyte and
platelet/multilineage findings; hepatic enzymes, bilirubin/liver injury and
coagulation; sodium/potassium, calcium/magnesium, glucose/lipids/albumin and
phosphate/urate/cell turnover. Treatment escalation and subsequent malignancy
join disease course; cardiac and thrombotic/cerebrovascular outcomes join one
cardiovascular category. Bone fracture joins acute-care endpoints in the
explicitly heterogeneous "Other acute events and care use" display group;
this grouping is pragmatic and does not imply a shared mechanism.
Composite endpoints are assigned once for display even though they share
component events with other groups. Thus group medians describe a set of
correlated endpoint AUROCs, not independent biological effects.

Points to vet with the clinical team before treating this as publication
taxonomy:

- `severe_cytopenia` is a multilineage composite, displayed alongside platelet
  outcomes for compactness; it could instead be its own composite category.
- `LDH_increased` is a nonspecific cell-turnover marker, provisionally grouped
  with phosphate/urate and tumor lysis.
- `infections_all_definitions` spans multiple infection mechanisms but is
  displayed with antimicrobial proxies; moving it to the severe-infection
  group would be equally defensible.
- The bars in
  the prototype are between-endpoint IQRs, not confidence intervals.

No group labels should be used to imply treatment-attributable toxicity or
source-specific clinical adjudication. The underlying outcome-definition
review and final-data audit remain separate tasks.
