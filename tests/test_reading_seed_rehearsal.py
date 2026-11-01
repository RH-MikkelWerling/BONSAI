"""Replica pairing preserves seed identity without confusing analysis groups."""
import pandas as pd
import pytest

from opera.run.audit_reading_seed_rehearsal import analyse, pairing_context


def predictions():
    rows = []
    for model, seed in (("J_OA_OWN_L", 42), ("J_OA_OWN_s43_L", 43), ("J_L", 42)):
        for i in range(20):
            rows.append(dict(model=model, seed=seed, split="tuning", cohort="DLBCL",
                             subject_id=str(i), outcome="infection", horizon_days=365,
                             time_days=30 if i < 10 else 500, event=int(i < 10),
                             label=int(i < 10), known=True, data_contract="fixture",
                             probability=(.8 if i < 10 else .2) if model != "J_L" else .5))
    return pd.DataFrame(rows)


def test_replicas_pair_and_identical_predictions_have_zero_gain():
    frame = predictions()
    original = frame.copy(deep=True)
    context, identities = pairing_context(frame)
    assert identities["J_OA_OWN_s43_L"] == 43
    assert set(context.seed) == {0}
    assert set(context.training_seed) == {42, 43}
    metadata = pd.DataFrame([dict(cohort="DLBCL", outcome="infection", horizon_days=365)])
    tables, _ = analyse(frame, metadata, replicates=20)
    paired = tables["contrasts"].query("contrast == 'OAOWN_NF' and metric == 'auc'")
    assert len(paired) == 1 and paired.iloc[0].gain == 0
    assert paired.iloc[0].left_training_seed == 42 and paired.iloc[0].right_training_seed == 43
    summaries = tables["effect_summaries"].query("contrast == 'OAOWN_NF' and metric == 'auc'")
    assert len(summaries) and set(summaries.weighting) == {"equal_cell", "disease_balanced"}
    assert summaries.gain.eq(0).all()
    assert set(tables["model_auc"].training_seed) == {42, 43}
    pd.testing.assert_frame_equal(frame, original)


def test_seed_rehearsal_refuses_test_rows_or_unmatched_patients():
    frame = predictions()
    with pytest.raises(ValueError, match="tuning"):
        pairing_context(frame.assign(split="held_out"))
    with pytest.raises(ValueError, match="rosters differ"):
        pairing_context(frame.iloc[1:])


def test_replica_alias_cannot_silently_combine_training_seeds():
    frame = predictions()
    frame.loc[0, "seed"] = 44
    with pytest.raises(ValueError, match="exactly one"):
        pairing_context(frame)
