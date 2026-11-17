import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from opera.evaluation.distributed_analysis import save_json, save_pickle
from opera.evaluation.identity_result_recovery import completed_identity_sources, recover


def test_exact_contrasts_recovered_without_other_outputs(tmp_path):
    save_json(tmp_path/'queue.json', {'total':2})
    expected=[]
    for n in range(2):
        rows=[dict(split='held_out', seed=42, metric='auc', gain=.01+n,
                   lower=-.1, upper=.2, contrast='P1', cohort='a', outcome='o')]
        expected.extend(rows)
        save_pickle(tmp_path/f'batch_{n:05d}.result_INTERNAL.pkl',
                    ([], rows, [{'gain_draws':np.arange(1000)}], [], [], []))
    table=recover(tmp_path)
    destination=tmp_path/'published.csv'
    table.to_csv(destination,index=False)
    assert len(table)==2
    assert_frame_equal(pd.read_csv(destination),pd.DataFrame(expected))
    # Reuse validates the marker without reloading result pickles.
    from unittest.mock import patch
    with patch('opera.evaluation.identity_result_recovery.load_pickle',side_effect=AssertionError):
        assert len(recover(tmp_path))==2
    (tmp_path/'batch_00000.result_INTERNAL.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        recover(tmp_path)


def test_incomplete_results_refused(tmp_path):
    save_json(tmp_path/'queue.json', {'total':1})
    assert completed_identity_sources(tmp_path) is None
    with pytest.raises(ValueError,match='Every identity batch'):
        recover(tmp_path)
