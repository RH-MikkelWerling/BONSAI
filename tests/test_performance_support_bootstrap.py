import numpy as np
import pandas as pd
from opera.evaluation.secondary_analysis import patient_bootstrap, paired_analysis


def frame():
    rows=[]
    for model in ['J', 'R']:
        for sid in range(12):
            rows.append(dict(split='tuning', seed=42, cohort='A', outcome='x', horizon_days=365,
                             model=model, subject_id=str(sid), data_contract='fixture',
                             label=float(sid<5), probability=(.8 if sid<5 else .2) if model=='R' else .5))
    return pd.DataFrame(rows)


def test_support_rule_retains_coverage_but_does_not_bootstrap_small_cells():
    data=frame()
    small=data.assign(outcome='small', label=lambda f: (f.subject_id=='0').astype(float))
    result=paired_analysis(pd.concat([data,small]), {'R1':'gain(R,J)'}, replicates=30,
                           min_cases=5,min_controls=5,patient_gain_contrasts=[])
    assert result['estimates'].query("outcome=='small'").estimate.isna().all()
    assert result['contrasts'].query("outcome=='small'").gain.isna().all()
    assert result['contrast_coverage'].query("outcome=='small'").status.eq('insufficient_cases_controls').all()
    assert result['estimates'].query("outcome=='x' and metric=='auc'").estimate.notna().all()


def test_positive_patient_weights_preserve_supported_auc_draws_and_are_shared():
    data=frame()
    draws=patient_bootstrap(data,100,23,method='multiplier')
    mapping,weights=draws[('tuning',42,'A')]
    assert (weights>0).all()
    np.testing.assert_allclose(weights.sum(axis=1),12)
    again=patient_bootstrap(data,100,23,method='multiplier')
    np.testing.assert_array_equal(weights,again[('tuning',42,'A')][1])
    result=paired_analysis(data,{'R1':'gain(R,J)'},draws=draws,min_cases=5,min_controls=5,
                           patient_gain_contrasts=[])
    auc=result['contrasts'].query("metric=='auc'").iloc[0]
    assert auc.n_valid_bootstrap==100
    assert auc.gain==.5
