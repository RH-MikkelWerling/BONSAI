import pandas as pd

from bonsai.functional.features import compute_abspos
from opera.functional.outcomes import attach_prediction_censor_abspos


def test_attach_prediction_censor_abspos_uses_index_date_not_followup_censor_date():
    outcomes = pd.DataFrame(
        {
            "subject_id": [1],
            "index_date": [pd.Timestamp("2020-01-01")],
            "censor_date": [pd.Timestamp("2021-01-01")],
        }
    )

    with_abspos = attach_prediction_censor_abspos(outcomes)

    assert with_abspos.loc[0, "censor_abspos"] == compute_abspos(outcomes["index_date"]).iloc[0]
    assert with_abspos.loc[0, "censor_abspos"] != compute_abspos(outcomes["censor_date"]).iloc[0]
