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


def test_binary_metrics_at_frozen_group_boundaries():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import matthews_corrcoef
    from opera.run.render_clinical_score_panels import classification_at_boundaries
    data = pd.DataFrame(dict(subject_id=list('abcdefgh'),
        model_risk=[.1,.2,.3,.4,.6,.7,.8,.9],
        ipi_group=['Low']*3+['Intermediate']*2+['High']*3,
        time_days=[500,500,20,500,20,20,500,20], event=[0,0,1,0,1,1,0,1]))
    spec = dict(id='score',model='R_L',horizon_days=365,threshold_split='train',
                group_order=['Low','Intermediate','High'])
    rows = pd.DataFrame(classification_at_boundaries(data,spec,[.3,.7],'tuning'))
    assert len(rows) == 8
    row = rows.loc[(rows.system=='model') & (rows.boundary=='High') & (rows.estimand=='known_status')].iloc[0]
    assert (row.tp,row.fp,row.fn,row.tn) == (2,1,2,3)
    assert row.recall == .5
    assert row.precision == 2/3
    assert np.isclose(row.mcc,matthews_corrcoef([0,0,1,0,1,1,0,1],[0,0,0,0,0,1,1,1]))
    assert set(rows.threshold_split) == {'train'}
    data.loc[0,['time_days','event']] = [10,0]
    data.loc[1,['time_days','event']] = [20,2]
    rows = pd.DataFrame(classification_at_boundaries(data,spec,[.3,.7],'held_out'))
    ordinary = rows.loc[rows.estimand=='known_status']
    assert ordinary.n_known.eq(7).all()
    assert (ordinary[['tp','fp','fn','tn']].sum(axis=1)==7).all()


def test_training_native_predictions_require_explicit_opt_in_and_keep_cif_weights(tmp_path):
    from opera.run.analyse_locked_predictions import read_artifact
    path = tmp_path / 'train.parquet'
    pd.DataFrame(dict(subject_id=['a', 'b', 'c', 'd'], cohort='DLBCL',
                      split='train', outcome='overall_survival',
                      time_days=[10, 20, 500, 500], event=[0, 1, 0, 0],
                      cif_365d=[.1, .8, .2, .3])).to_parquet(path)
    spec = dict(path=str(path), format='direct_cif', model='R_L', seed=42,
                split='train', data_contract='registered-release')
    with pytest.raises(ValueError, match='tuning/held_out splits'):
        read_artifact(spec, estimand='cif_ipcw')
    frame, _ = read_artifact(spec, estimand='cif_ipcw', allow_training=True)
    assert set(frame.split) == {'train'}
    assert set(frame.data_contract) == {'registered-release'}
    assert frame.set_index('subject_id').loc['a', 'ipcw_weight'] == 0
    assert frame.set_index('subject_id').loc['b', 'ipcw_weight'] == pytest.approx(4/3)


def test_clinical_renderer_runs_native_train_and_tuning_exports_end_to_end(tmp_path, monkeypatch):
    """Exercise actual adaptation, thresholds, metrics and plots, without mocked readers."""
    import json
    import numpy as np
    import yaml
    from opera.run import render_clinical_score_panels as renderer
    from opera.run.clinical_figure_provenance import NATIVE_FOUR, verify_clinical_render

    rng = np.random.default_rng(431)
    metadata, artifacts = [], []
    for split, n in [('train', 600), ('tuning', 240)]:
        risk = np.linspace(.01, .95, n)
        groups = np.repeat(NATIVE_FOUR, n // 4)
        event = rng.binomial(1, .02 + .65 * risk)
        ids = [f'{split}_{i}' for i in range(n)]
        path = tmp_path / f'{split}.parquet'
        # The native exporter does not include data_contract/model/seed.
        pd.DataFrame(dict(subject_id=ids, split=split, outcome='overall_survival',
                          time_days=np.where(event == 1, 100, 500), event=event,
                          cif_365d=risk)).to_parquet(path)
        metadata.append(pd.DataFrame(dict(subject_id=ids, cohort_fine='DLBCL',
            age_at_index=np.tile([50, 65, 80], n // 3), nccn_ipi_group=groups,
            nccn_ipi_score=np.repeat([0, 2, 4, 7], n // 4))))
        artifacts.append(dict(path=str(path), model='R_L', seed=42, split=split,
                              format='direct_cif', cohort_from_metadata=True))
    artifacts[1]['data_contract'] = 'registered-release'
    meta = tmp_path / 'metadata.parquet'
    pd.concat(metadata).to_parquet(meta)
    analysis = tmp_path / 'analysis.yaml'
    analysis.write_text(yaml.safe_dump(dict(patient_metadata=str(meta),
        artifacts=[artifacts[1]], clinical_threshold_artifacts=[artifacts[0]])))
    score = dict(id='nccn_ipi', model='R_L', seed=42, cohorts=['DLBCL'],
        outcome='overall_survival', outcome_label='Overall survival', horizon_days=365,
        score_label='NCCN-IPI', score_column='nccn_ipi_score', score_high_threshold=6,
        group_column='nccn_ipi_group', group_order=NATIVE_FOUR,
        group_map={g: g for g in NATIVE_FOUR}, age_column='age_at_index',
        score_timing='Synthetic baseline', threshold_split='train',
        cutoff_rule='risk_matched_low', death_in_endpoint=True,
        manuscript_figures=True, lock_provenance='Synthetic regression test',
        applicability_review='Synthetic regression test')
    config = tmp_path / 'scores.yaml'
    config.write_text(yaml.safe_dump(dict(scores=[score])))
    output = tmp_path / 'clinical_scores'
    monkeypatch.setenv('OPERA_PAPER_FIGURE_MARK', 'TUNING SET — not test results')
    monkeypatch.setattr('sys.argv', ['render', '--analysis-config', str(analysis),
        '--score-config', str(config), '--output-dir', str(output), '--split', 'tuning', '--execute'])
    renderer.main()
    inventory = pd.read_csv(output / 'score_inventory.csv')
    assert inventory.status.tolist() == ['rendered']
    assert inventory.threshold_n.tolist() == [600]
    assert inventory.threshold_split.tolist() == ['train']
    assert inventory.eligible_n.tolist() == [240]
    metrics = pd.read_csv(output / 'classification_at_thresholds.csv')
    assert set(metrics.split) == {'tuning'}
    assert set(metrics.boundary) == set(NATIVE_FOUR[1:])
    assert {'recall', 'precision', 'specificity', 'mcc'} <= set(metrics)
    assert (output / 'fig_clinical_score.pdf').is_file()
    receipt = json.loads((output / 'report_manifest.json').read_text())
    assert set(receipt['prediction_data_contracts'].values()) == {'registered-release'}
    assert verify_clinical_render(output, expected_model='R_L')['valid']
