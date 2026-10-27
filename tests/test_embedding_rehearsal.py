import numpy as np
import pandas as pd
import pytest

from opera.run.build_paper_embedding_tables import project


def _frame(splits):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"subject_id": [f"p{i}" for i in range(len(splits))],
                         "split": splits,
                         "embedding": list(rng.normal(size=(len(splits), 8)))})


def test_rehearsal_shows_half_the_tuning_patients():
    frame = _frame(["tuning"] * 200)
    shown = project(frame, method="pca", rehearse=True)
    assert 60 < len(shown) < 140
    again = project(frame, method="pca", rehearse=True)
    assert shown.subject_id.tolist() == again.subject_id.tolist()


def test_rehearsal_refuses_held_out_exports():
    with pytest.raises(ValueError, match="tuning-only"):
        project(_frame(["tuning"] * 50 + ["held_out"] * 50), method="pca", rehearse=True)
    with pytest.raises(ValueError, match="held-out"):
        project(_frame(["tuning"] * 50), method="pca")
