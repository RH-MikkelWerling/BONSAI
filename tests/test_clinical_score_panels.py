import pandas as pd
import pytest
from opera.run.render_clinical_score_panels import score_patients


def test_score_population_and_preindex_availability_are_explicit():
    predictions = pd.DataFrame(
        dict(
            subject_id=["a", "b", "c"],
            model=["J_L"] * 3,
            seed=[42] * 3,
            split=["held_out"] * 3,
            outcome=["failure"] * 3,
            horizon_days=[730] * 3,
            cohort=["DLBCL"] * 3,
            probability=[0.1, 0.2, 0.3],
            time_days=[800] * 3,
            event=[0] * 3,
        )
    )
    metadata = pd.DataFrame(
        dict(
            subject_id=["a", "b", "c"],
            score=[1, 2, 3],
            group=["L", "H", "L"],
            age=[50, 60, 70],
            score_date=["2020-01-01", "2020-02-01", None],
            index_date=["2020-01-02"] * 3,
        )
    )
    spec = dict(
        model="J_L",
        seed=42,
        outcome="failure",
        horizon_days=730,
        cohorts=["DLBCL"],
        score_column="score",
        group_column="group",
        age_column="age",
        score_date_column="score_date",
        index_date_column="index_date",
        group_map={"L": "Low", "H": "High"},
    )
    frame, audit = score_patients(predictions, metadata, spec, "held_out")
    assert frame.subject_id.tolist() == ["a"]
    assert audit == dict(
        eligible_n=3, score_available_at_index_n=1, unavailable_or_late_n=2
    )
    with pytest.raises(ValueError, match="Duplicate"):
        score_patients(
            pd.concat([predictions, predictions]), metadata, spec, "held_out"
        )
    spec["group_map"] = {"H": "High"}
    with pytest.raises(ValueError, match="Unmapped"):
        score_patients(predictions, metadata, spec, "held_out")


def test_clinical_score_event_on_horizon_is_positive():
    from opera.visualization.paper_figures import _known_status

    frame = pd.DataFrame({"time_days": [730, 729, 730], "event": [1, 0, 2]})
    known, labels = _known_status(frame, 730)
    assert known.index.tolist() == [0, 2]
    assert labels.tolist() == [1, 0]
