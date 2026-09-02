import numpy as np
import torch

from opera.run.evaluate import _batch_numpy_vector


def test_batch_numpy_vector_preserves_regular_batch():
    result = _batch_numpy_vector(torch.tensor([[1.0], [2.0]]))

    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


def test_batch_numpy_vector_keeps_singleton_batch_concatenation_safe():
    singleton = _batch_numpy_vector(torch.tensor([[1.0]]))
    regular = _batch_numpy_vector(torch.tensor([[2.0], [3.0]]))

    assert singleton.shape == (1,)
    np.testing.assert_array_equal(
        np.concatenate([regular, singleton]), np.array([2.0, 3.0, 1.0])
    )
