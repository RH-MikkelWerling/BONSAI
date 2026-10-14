# First-line final-cohort review (INTERNAL; 2026-10-02)

## Decision boundary

The current mapped export is not yet a defensible final paper cohort. Freeze
the new MEDS release and its vocabulary contract, then build a coherent,
versioned first-line release and all index-dependent products from that release.
Do not overwrite the old artifacts. Pretraining depends on the MEDS release,
but not on the first-line cohort; downstream supervised training does depend on
the final cohort/index/outcome/split release.

## Evidence from the current release comparison

Old/new mapped counts are 37,494/37,482 with 37,481 shared patients. Among
shared patients, 280 first-line dates, 431 first-line sources, 173 primary arms,
29 fine cohorts, and 15 grouped cohorts differ. This cannot be treated as a
12-person-only change. Current CLL seeds are associated with 193 changed dates;
no current seed is associated with another 84. The historical CLL seed file or
its hash has not been recovered, so the exact upstream change remains unproven.

## Code-review findings requiring action before release

1. **High: seed provenance and disease alignment.** The script loads line-1
   CLL/MM seed Parquets, retains the earliest seed per patient regardless of
   disease, and joins only `patient_id`/`seed_date` to treatment episodes. The
   selected seed can therefore move an index for a different disease. The
   existing diagnostic found 307 patients labelled outside CLL/SLL among 4,247
   with selected CLL seed, and 86 outside MM among 8,568 with selected MM seed.
   These are flags for adjudication, not proof that each case is wrong; a person
   can have more than one malignancy or transform. Inspect the seed-generation
   code and patient-level discordance on the secure server. A disease-specific
   matching rule requires a prospective decision and a sensitivity comparison.
2. **High: exported branch is not the later `FROZEN` branch.** The mapped CSV
   is built with CD20 monotherapy acceptance on, whereas a later section resets
   it off, writes separate RDS objects, and still exports the earlier mapped
   object. The exploratory RKKP fallback is later again. Replace this with one
   explicit production driver and a policy manifest; verify that its date,
   cohort, arm, expanded features and counts all refer to the same object.
3. **High: external mutable inputs.** `patient`, `t_dalycare_diagnoses`, ICD-10
   code lists and later RKKP tables come from an external loader/SQL. Treatment
   and CLL/MM line tables come from external Parquets. The MM 100-line file is
   loaded but immediately overwritten by a 5-line file. Record the effective
   source only, and obtain the generator scripts, executed SQL/view definitions,
   source commit, extraction timestamp, snapshot/backup ID and checksums. A
   live table name alone is not a frozen input.
4. **Medium-high: cohort loss is heterogeneous.** Pass 1 left 1,188 indexed
   patients unclassified; pass 2 assigned 198, leaving 990. Some may lack a
   specific diagnosis, some may have only excluded/unspecific codes, some may
   have diagnosis outside the window, and some may have treatment for another
   indication. They cannot all be rescued using treatment arm labels. The
   script now includes `reason` in aggregate audit breakdowns for the existing
   pass-1 unclassified stage. Review those counts plus unmapped diagnosis-code
   and treatment-indication distributions on the server before choosing any
   new rescue. Arm-based labels would be lower-resolution/less independent and
   should be retained only as a separately flagged sensitivity branch.
5. **Medium-high: retrospective diagnosis assignment.** Pass 2 looks for a
   confirmable specific diagnosis anywhere in history, then uses a 90-day
   forward grace from treatment. This may be clinically useful for coding lag,
   but a later diagnosis can alter cohort and index eligibility. Audit the
   diagnosis-to-index lag and transitions by disease, and explicitly define
   whether the primary cohort is contemporaneous or retrospectively adjudicated.
6. **Medium: therapy/indication gate.** RKKP-origin records can pass the
   indication gate even if an exclusion indicator is present. Unknown
   indications are kept. `therapy_like` is calculated twice, with the first
   calculation overwritten. A prophylaxis-suspect episode requires both three
   dates and two codes; a comment questions this policy. The CD20 suspect text
   contains `cd29 antibody`, apparently a typo. Test each branch with examples
   and count patients uniquely affected before making clinical changes.
7. **Medium: diagnosis mapping and tie-breaking.** Mapping uses prefix/regex
   matching with overlapping groups and deduplication. Disease assignment
   favors aggressive labels and then source priority, recency and specificity;
   tied rows can depend on input ordering. Derived RT/transformed-FL labels
   use earliest *mapped* dates as if confirmed; this needs validation against
   actual confirmation chronology. Review all ICD-code overlaps and priority
   ties, especially NMZL/SMZL and CLL/SLL.
8. **Low-medium: exploratory script hygiene.** `sample_n(100)` can fail if
   fewer than 100 exclusions; the script contains later unguarded exploratory
   work and relative paths dependent on the original working directory.
   Remove these from the production driver. The counts-only/candidate wrappers
   currently avoid the later sections, but do not resolve the clinical policy.

## Controlled rebuild order

1. Snapshot the exact new MEDS dataset/manifest and source code. Locate the
   scripts/SQL that generated CLL/MM line seeds and cleaned treatment records;
   capture the external loader and ICD-10 definitions. Resolve the original
   tabular feature generator, not just this repo's feature-filtering code.
2. Produce an aggregate and secure patient-level review of seed discordance,
   the 990 unclassified cases, pass-2 temporal lag, arm-rescue eligibility,
   and branch sensitivity (seed on/off, disease-aligned seed, CD20/RKKP policy).
   Clinical adjudication should sample discordant and changed-date cases.
3. Approve a single first-line policy. Build/test one production driver with
   explicit inputs, deterministic tie-breaks, validation assertions and a
   release manifest. Run into a new immutable versioned directory. Compare its
   patient IDs, dates, cohort/arm labels and exclusions with the old release.
4. From that **one** index release, regenerate second-line ascertainment,
   outcomes (including vetted panel/TLS), observation/follow-up/splits, tabular
   feature matrices and sequence fine-tuning inputs. Verify index-time leakage
   rules and all row/ID/key counts. Regenerate Table 1/CONSORT and models from
   the coherent release. Pretraining on frozen MEDS may proceed in parallel
   after the tokenizer/vocabulary contract is verified.

No clinical-policy or patient-level change is made by this review. The only
instrumentation change is an aggregate `reason` breakdown in the existing
first-line count output; it is INTERNAL, not disclosure-cleared.
