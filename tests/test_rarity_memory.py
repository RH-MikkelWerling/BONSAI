import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from opera.run.analyse_locked_predictions import rarity_inputs


def test_chunked_rarity_matches_previous_with_nonunique_index():
    frame = pd.DataFrame(dict(metric=['auc', 'brier', 'other', 'ipa', 'log_loss'],
        cohort=['a', 'b', 'c', 'a', 'b'], outcome=['o']*5,
        horizon_days=[30,60,90,180,365], contrast=['P1']*5,
        gain=[.1,.2,.3,.4,.5], n_cases=[1,2,3,np.nan,5],
        n_controls=[4,3,2,1,np.nan], extra=['x']*5), index=[2,2,1,0,0])
    expected=frame.loc[frame.metric.isin(['auc','ipa','brier','log_loss'])].copy()
    expected['metric']=expected.metric.replace({'auc':'auroc','brier':'brier_score'})
    expected['difference']=expected.gain
    expected['minority_class']=expected[['n_cases','n_controls']].min(axis=1)
    expected['cell_id']=expected[['cohort','outcome','horizon_days','contrast']].astype(str).agg('::'.join,axis=1)
    assert_frame_equal(rarity_inputs(frame,chunk_rows=2),expected)
    assert_frame_equal(frame, frame.copy())
