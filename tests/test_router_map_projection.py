import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from opera.run.build_paper_embedding_tables import project, calibration_map_inputs
from opera.visualization.paper_figures import figure_horizons
from opera.run.render_paper_figure_prototypes import synthetic_cells


def test_arrows_use_one_fitted_projection_and_reject_another_checkpoint(tmp_path):
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(30, 6))
    frame = pd.DataFrame({'subject_id': [str(i) for i in range(30)],
                          'split': ['tuning'] * 20 + ['held_out'] * 10,
                          'embedding': list(embeddings)})
    path = tmp_path / 'vectors.npz'
    path.with_suffix('.json').write_text('{"target_outcome": "x", "split": "held_out"}')
    np.savez(path, subject_ids=np.arange(30), shared=embeddings, adapted=embeddings)
    coords = project(frame, method='pca', routed_vectors={'x': path})
    np.testing.assert_allclose(coords[['x', 'y']], coords[['routed:x:x', 'routed:x:y']])
    predictions = pd.DataFrame({'subject_id': range(20, 30), 'outcome': 'x',
                                'time_days': 400., 'event': 0, 'cif_365d': .1})
    _, payload = calibration_map_inputs(coords, predictions, ['x'])
    assert np.asarray(next(iter(payload.values()))[3]).shape == (10, 2)
    np.savez(path, subject_ids=np.arange(30), shared=embeddings + .1, adapted=embeddings)
    with pytest.raises(ValueError, match='same checkpoint'):
        project(frame, method='pca', routed_vectors={'x': path})


def test_horizons_includes_router_and_accepts_single_panel():
    cells = synthetic_cells(np.random.default_rng(1))
    figure = figure_horizons(cells)
    assert len(figure.axes) == 4
    assert 'Routed' in figure.axes[-1].get_title(loc='left')
    plt.close(figure)
    figure = figure_horizons(cells, contrasts=('R1',))
    assert len(figure.axes) == 1
    plt.close(figure)


def test_calibration_map_rejects_conflicting_duplicates_and_empty_patient_overlap():
    coords = pd.DataFrame({'subject_id': ['1'], 'x': [0.], 'y': [0.]})
    predictions = pd.DataFrame({'subject_id': ['1', '1'], 'outcome': ['x', 'x'],
                                'time_days': [400., 400.], 'event': [0, 0],
                                'cif_365d': [.1, .9]})
    with pytest.raises(ValueError, match='Conflicting prediction rows'):
        calibration_map_inputs(coords, predictions, ['x'])
    with pytest.raises(ValueError, match='No common evaluated patients'):
        calibration_map_inputs(coords.assign(subject_id='2'), predictions.iloc[:1], ['x'])
