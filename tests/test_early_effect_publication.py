import pandas as pd
from pandas.testing import assert_frame_equal
from opera.run.analyse_locked_predictions import publish_main_effects


def test_early_table_separate_from_final_output(tmp_path):
    output=tmp_path/'results'
    table=pd.DataFrame({'contrast':['EXT_ADAPTATION'], 'gain':[0.01]})
    publish_main_effects(output,table)
    path=tmp_path/'results.progress/effect_summaries.csv'
    assert_frame_equal(pd.read_csv(path),table)
    assert not output.exists()
    table['gain']=0.02
    publish_main_effects(output,table)
    assert_frame_equal(pd.read_csv(path),table)
    assert list(path.parent.iterdir()) == [path]
