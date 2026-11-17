import pandas as pd
from opera.run.audit_external_meds_bins import replay, summarize


def test_replay_detects_wrong_external_bins_and_suffix(tmp_path):
    shard = tmp_path/'data/train/part.parquet'
    shard.parent.mkdir(parents=True)
    frame = pd.DataFrame({'code':['LABTEST//A//BIN_0','LABTEST//A//BIN_1','LABTEST//NEW'],
        'numeric_value':[0.,10.,5.], 'numeric_value_bin':[0.,1.,float('nan')]})
    frame.to_parquet(shard, index=False)
    metadata = pd.DataFrame({'code':['LABTEST//A'], 'numeric/bin_1':[0.],
        'numeric/bin_99':[10.], 'numeric/bin_edges':[[.5]]})
    files, counts = summarize(tmp_path)
    assert counts.loc['LABTEST//A','binned_events'] == 2
    report = replay(files, metadata, 100)
    assert report['sampled_numeric_labs'] == 3
    assert report['external_transform_mismatches'] == 0
    assert report['bin_suffix_mismatches'] == 0
    frame.loc[0,'numeric_value_bin'] = 1
    frame.to_parquet(shard, index=False)
    report = replay(files, metadata, 100)
    assert report['external_transform_mismatches'] == 1
    assert report['bin_suffix_mismatches'] == 1


def test_intermediate_replay_distinguishes_post_binning_rounding(tmp_path):
    import numpy as np
    from opera.run.audit_external_meds_bins import intermediate_replays
    metadata = pd.DataFrame({'code':['LABTEST//A'], 'numeric/bin_1':[0.],
        'numeric/bin_99':[1.], 'numeric/bin_edges':[[0.7]]})
    before = tmp_path/'annotate_numeric_values/train/part.parquet'
    before.parent.mkdir(parents=True)
    pd.DataFrame({'code':['LABTEST//A'], 'numeric_value':[0.7],
                  'numeric_value_bin':[1]}).to_parquet(before, index=False)
    after = tmp_path/'join_numeric_bins/train/part.parquet'
    after.parent.mkdir(parents=True)
    pd.DataFrame({'code':['LABTEST//A//BIN_1'], 'numeric_value':np.array([0.7], dtype=np.float32),
                  'numeric_value_bin':[1]}).to_parquet(after, index=False)
    report = intermediate_replays(tmp_path, metadata, 10)
    assert report['annotate_numeric_values']['external_transform_mismatches'] == 0
    assert report['annotate_numeric_values']['bin_suffix_mismatches'] == 0
    assert report['join_numeric_bins']['external_transform_mismatches'] == 1
    assert report['join_numeric_bins']['bin_suffix_mismatches'] == 0
    assert report['annotate_numeric_values']['numeric_value_dtypes'] == ['double']
    assert report['join_numeric_bins']['numeric_value_dtypes'] == ['float']
