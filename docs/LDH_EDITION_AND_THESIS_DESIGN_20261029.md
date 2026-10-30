# Independent editorial and design iteration — 29 October 2026

The working manuscript and live thesis are preserved. The new source is in
`paper/ldh_edition/`, with separate article wrappers in `paper/`.

## Open these first

- [LDH editorial reading PDF](../paper/render/ldh_edition_20261029/opera_ldh_editorial.pdf)
- [Same narrative, single-column review PDF](../paper/render/ldh_edition_20261029/opera_ldh_review.pdf)
- [Thesis facing-page design proof](../paper/render/ldh_edition_20261029/thesis_interludes_preview.pdf)
- [Source and build instructions](../paper/ldh_edition/README.md)

## What changed in the article

The revised narrative asks whether to borrow, what information helps, and how
to combine shared and task-specific learning. The questions are enumerated
in the introduction. Methods retain the clinical landmark, split, estimand,
matched supervision design, clinical cutoff provenance, and evaluation rules.
Detailed implementation remains in the existing source supplement.

Results now give answers rather than a catalogue of contrast identifiers.
The available tuning estimates show disease-sharing gains, limited additional
outcome-supervision gain after disease pooling, modest correction gains, and
the current XGBoost advantage. The clinical passage uses the refreshed local
count reconciliation and does not assume low-risk expansion at matching
evaluation mortality. The Discussion follows those distinctions and does not
require selective routing or national pretraining to win.

The section-aware expanded count is approximately 2,700 main-text words, leaving
room for final findings within 3,500. The five-part Summary is approximately
250 words, with 27 main references. Final author confirmations remain explicit.
No new evidence search is claimed in Research in context.

## Layout and figures

The reading copy uses a restrained crimson accent, Times-family body type,
Helvetica-family headings, a study sidebar, superscript references, two-column
text, compact tables, and separate full-width figure pages. It is inspired by
clinical-journal presentation rather than an official publisher template.
The second copy supplies a more conventional single-column review layout.

Six figure pages were created from local aggregate snapshots:

1. Treatment landmark, temporal split, and the four supervision scopes.
2. Absolute performance and paired AUC/IPA contrasts against practical alternatives.
3. Available clinical counts and achieved operating points, with interim status clear.
4. Marginal and conditional disease/outcome supervision effects, with uncertainty.
5. Separately trained correction controls and support-dependent heterogeneity.
6. National-transfer design, with no synthetic performance estimates.

This reduces reliance on model taxonomies, embedding geometry, and intricate
source maps in the main account. Projected arrows, attribution atlases, and
source-pair diagnostics remain supplementary. The revised router is still
primary, with the original as a sensitivity comparison. No fit was selected
or retrained for this design iteration.

Visual iteration corrected clipped long model/contrast labels, title overflow,
clinical note spacing, and the correction legend's separation from the footer.
The small positive full-versus-own-only interval boundary is printed to four
decimal places so that rounding does not turn it into an apparent zero.
Figure 5 uses IQR specifically for variation across tasks and confidence
intervals for mean effects. No new bootstrap intervals were computed.

The broader computed-IPI population is kept distinct from DLBCL NCCN-IPI.
Calibration is not inferred from IPA. Conventional calibration curves and
native-category clinical survival figures must be replaced from the verified
rerun outputs. Their absence is stated rather than filled with synthetic data.

## What changed for the thesis

The [artist portfolio](https://www.petrapeterffy.com/) was examined as a visual
reference, including its Stanford Medicine work. Three original, procedural
vector treatments were made using the thesis's existing indigo, plum, ruby,
and rose palette:

- **Portrait:** a synthetic patient-like silhouette passing from discrete tiles
  into continuous contour lines, maintaining continuity with the existing cover.
- **Currents:** interlocking arcs and repeated lines, with discrete event fragments.
- **Quiet:** a smaller lower-page motif with substantial unprinted white space.

The proof places each on an actual even page facing a chapter opening. The
optional memoir package uses the existing `cleared` style and adds no pages.
Chapter fragments are explicitly specimens, not revisions of the thesis text.
White-background asset variants are provided. No artist image or patient data
enters the artwork.

My recommendation is to retain the current cover and use the quiet treatment
for selected synthesis transitions, with a larger portrait or currents piece
at a major part division. Filling every blank page would dilute their purpose.
The live thesis has not been edited.

## What remains before submission

This is an editable editorial edition, not a completed submission. Remaining
scientific inputs include the current clinical rerun, conventional calibration,
national joint/OPERA/adaptation and label-efficiency results, restrictions and
IG refresh, and the reserved temporal evaluation after lock. Remaining author
inputs include funding, contributor/data-verification roles, interests, patient
involvement, preferred degrees, corresponding-author details, code archive,
access arrangements, and the verified AI-assistance declaration.

The documented evidence search and final separately numbered journal appendix
also remain. The existing source supplement is retained, with no endpoint or
code-definition changes. Frozen uncertainty and support rules are unchanged.

Build receipts identify aggregate input and PDF hashes and verify that the
original manuscript sources remain unchanged. The final text and figure count
can be revised once the pending findings are available, without changing the
live scientific pipeline during the server rerun.

The [30 October close-edit handoff](LDH_MORNING_REVIEW_20261030.md) records the
subsequent wording and methodological clarifications. Both article PDFs have
been rebuilt from that revised text.
