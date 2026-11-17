from opera.run.audit_clinical_endpoint_config import audit_specs


def spec():
    return dict(id="iss_native", score_label="ISS", model="J_OA_OWN_L", cohorts=["MM"],
                endpoint_role="native", outcome="overall_survival", original_endpoint="Overall survival",
                horizons_days=[730, 365], threshold_split="train", cutoff_rule="cohort_size_matched",
                applicability_review="Reviewed: ISS in MM", lock_provenance="Dated definition review")


def test_audit_distinguishes_an_explicit_endpoint_from_a_template_fallback():
    valid = spec()
    fallback = {**valid, "id": "unmapped", "endpoint_role": None, "original_endpoint": None}
    result = audit_specs([valid, fallback], expected_model="J_OA_OWN_L", outcomes={"overall_survival"})
    assert result["scores"][0]["status"] == "configuration_complete"
    assert result["status"] == "needs_resolution"
    assert "missing_explicit_native_or_proxy_endpoint_mapping" in result["scores"][1]["issues"]


def test_audit_flags_wrong_model_unknown_outcome_and_unreviewed_population():
    invalid = {**spec(), "model": "J_L", "outcome": "unregistered", "applicability_review": None}
    row = audit_specs([invalid], expected_model="J_OA_OWN_L", outcomes={"overall_survival"})["scores"][0]
    assert set(row["issues"]) == {"model_differs_from_selected_primary",
                                 "outcome_not_in_approved_panel", "population_applicability_review_not_recorded"}


def test_audit_refuses_an_empty_roster_and_unresolved_horizons():
    assert audit_specs([], expected_model="R_E", outcomes=set())["status"] == "needs_resolution"
    row = audit_specs([{**spec(), "horizons_days": [None]}], expected_model="J_OA_OWN_L",
                      outcomes={"overall_survival"})["scores"][0]
    assert row["issues"] == ["unresolved_or_unregistered_horizon"]


def test_audit_preserves_proxy_disclosure_without_resolving_mapping():
    pending = {**spec(), "outcome": None, "endpoint_match": "proxy_pending_mapping",
               "endpoint_proxy_note": "Next treatment is not directly observed progression."}
    row = audit_specs([pending], expected_model="J_OA_OWN_L", outcomes={"overall_survival"})["scores"][0]
    assert row["endpoint_match"] == "proxy_pending_mapping"
    assert row["endpoint_proxy_note"] == pending["endpoint_proxy_note"]
    assert "unresolved_outcome" in row["issues"]
