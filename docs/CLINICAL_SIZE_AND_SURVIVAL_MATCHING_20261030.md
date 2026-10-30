# Clinical comparisons: same size or same survival

30 October 2026. Protocol candidate. No active server recipe, test result or
locked clinical definition has been changed.

**Implementation follow-up:** the explicit cohort-size/KM refresh mode is now
available, with paired comparisons, independent classifier metrics and
aggregate-only packaging. See [the refresh instructions](CLINICAL_AND_RECORD_GAP_REFRESH_20261030.md).
The active server recipes and the final scientific lock remain unchanged.

## Recommendation

Use a prespecified **size-matched risk-stratification benchmark** as the primary
grouped comparison. Retain development survival-matched Low identification as
a complementary clinical-count analysis. These ask different questions and
should not be forced into one set of cutoffs.

This advances the earlier low-anchored proposal, rather than registering that
candidate as the new default. The existing `size_matched` renderer option
uses development quantiles followed by frozen numeric cutoffs. It does not
currently implement exact within-evaluation size matching. The candidate
`fit_low_anchored_groups` uses IPCW event-risk matching, not direct KM matching.
Neither should be described as already implementing the proposal below.

## Primary: same number of patients, better prognosis

For each index, use the same eligible, score-available patients for both
systems, including patients subsequently censored. Count each native index
category using baseline information. Order these patients by the model's
predicted risk for the declared outcome and horizon. Assign approximately the
same number to each ordered model stratum, keeping tied scores together and
reporting deviations from the requested counts.

No observed survival times or future event labels select the boundaries.
Then estimate survival for the two sets of strata. For all-cause mortality,
the primary group comparison is the difference in one-year KM survival, with
paired uncertainty. For example, the question is whether the model can
identify the same number of Low patients with higher survival, and the same
number of High patients with lower survival. The latter means enrichment for
poor prognosis, not worse model performance.

Four native categories give four model strata without inventing intermediate
event-risk targets. A missing native category remains empty or explicitly
unavailable. Tiny native groups do not become precise merely because the
model matches their sizes. Do not merge them after seeing survival outcomes.

The within-evaluation matching rule must be declared before opening test
outcomes. It uses predictions and baseline index groups in the evaluation
cohort, but no outcome-based threshold optimisation. It is a **rank-based
benchmark**, not validation of a fixed numeric threshold for use on a single
new patient. The distinction matters: exact test group-size matching and
unchanged development numeric thresholds are generally different analyses.

For bootstrap uncertainty of this cohort-based rule, resample patients jointly
for both systems and repeat the baseline-only size matching in each replicate.
Keep duplicated tied scores together. Group counts can differ slightly from
their targets when ties prevent exact allocation. This evaluates uncertainty
of the prespecified rank-based comparison. A bootstrap conditional on fixed
observed memberships answers a different conditional question and must be
labelled accordingly.

The comparison is descriptive prognostic stratification. It does not establish
that changing treatment according to those categories benefits patients.

## Complementary: more patients at comparable one-year survival

In a declared development population, estimate native Low one-year survival.
Search attainable model cutoffs for the largest supported Low group with KM
survival at least that target. Use all follow-up records and KM directly if
the criterion is specifically KM survival, rather than substituting a
known-status event fraction. Require sufficient horizon follow-up and report
uncertainty of the native reference and selected model group.

For this binary Low-identification question, restrictions introduced only to
populate three arbitrary remaining categories need not constrain the optimum.
A full K-category low-anchored partition is a separate descriptive option.
Its Low group is largest subject to those additional partition constraints,
not necessarily the largest Low group for the binary clinical question.

Freeze the selected numeric cutoff before testing. In the test cohort, report
the number classified Low, one-year survival, death counts and the paired
survival difference with uncertainty. Development matching does not establish
equal test survival. Failure to reject a difference also does not establish
equivalence. Any formal noninferiority claim needs a justified margin declared
before testing and enough precision to assess it.

Selecting the cutoff by searching test KM estimates until they match would
use test outcomes for development. Do not do that for a claimed frozen
classifier. A test-cohort risk/coverage curve can be an exploratory performance
display, but it is not the same validation claim.

Bootstrap the complete development selection to expose instability. This
uses the existing predictions and requires no predictive model retraining.
It cannot create events or remove prior model-selection use of tuning.

## Endpoint and metric choices

KM survival is appropriate for all-cause death. For nonfatal outcomes with
death as a competing event, compare Aalen-Johansen cumulative incidence
instead. A death-inclusive composite can use event-free KM survival. Match
the estimator, endpoint and horizon between the index and the model.

The common one-year comparison should be distinguished from each index's
original endpoint and validation horizon. Neither grouping approach replaces
continuous discrimination, raw calibration or probability-error evaluation.

Continue reporting ordinary known-status and censoring-weighted binary metrics
at the declared category boundaries: recall, precision, specificity, NPV, MCC,
accuracy, F1 and confusion counts. The matched-specificity binary comparison
remains separate from the model's High stratum. For cohort-based size matching,
state that the boundary is a rank rule rather than a development-frozen number.

## Relation to established practice

Prognostic score quantiles and KM curves are established ways to display risk
stratification. [Royston and Altman](https://link.springer.com/article/10.1186/1471-2288-13-33)
also discuss the information lost through categorisation and the need for
broader validation. Their paper does not specifically prescribe our proposed
native-index-size matching scheme. That scheme is our benchmark design.

[TRIPOD-Cluster's explanation](https://www.bmj.com/content/380/bmj-2022-071058)
requires the grouping choices to be explained and distinguishes statistical
boundaries from clinically justified decision thresholds. Those requirements
support explicit benchmark labels rather than implying that our strata are
new established clinical instruments.

The appropriate use of competing-risk estimators is discussed in
[BMJ's competing-risk validation guide](https://www.bmj.com/content/377/bmj-2021-069249).

## Decisions needed before implementation and test lock

1. Whether the primary grouped comparison uses within-evaluation rank matching
   or development-frozen numeric cutoffs. The former gives directly comparable
   sizes; the latter tests transport of an individual classification threshold.
2. Whether direct KM-matched expansion is a complementary analysis, with its
   development source, support and instability checks specified.
3. Which outcomes/horizons and score populations are primary, plus the native
   endpoint sensitivity analyses and paired uncertainty procedure.

These are analysis-definition decisions. They do not require model retraining,
new outcome definitions, or repetition of completed recent-record inference.
