"""Disk-backed bootstrap storage preserves draws and the public NPZ schema."""
import numpy as np
import pandas as pd
from opera.evaluation.secondary_analysis import disk_backed_draw_matrix, _spill_batch_draws


def test_disk_matrix_preserves_float64_nan_and_compressed_export(tmp_path):
    values = [np.array([.123456789012345, np.nan, -2.]), np.array([4., 5., 6.])]
    mapped = disk_backed_draw_matrix(pd.DataFrame({'gain_draws': values}), 'gain_draws')
    assert isinstance(mapped, np.memmap)
    assert mapped.dtype == np.float64
    np.testing.assert_array_equal(mapped, np.stack(values))
    destination = tmp_path / 'draws.npz'
    np.savez_compressed(destination, gains=mapped)
    with np.load(destination) as restored:
        np.testing.assert_array_equal(restored['gains'], np.stack(values))


def test_large_batch_spills_both_draw_tables_without_changing_values():
    rng = np.random.default_rng(4)
    original = rng.normal(size=(150, 1000))
    original[0, 0] = np.nan
    part = ([], [], [{'gain_draws': v.copy()} for v in original], [], [],
            [{'estimate_draws': v.copy()} for v in original])
    _spill_batch_draws(part)
    for index, column in ((2, 'gain_draws'), (5, 'estimate_draws')):
        assert isinstance(part[index][0][column], np.memmap)
        np.testing.assert_array_equal(np.stack([r[column] for r in part[index]]), original)
