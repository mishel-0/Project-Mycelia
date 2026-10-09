import numpy as np
import pytest

from tools.conventional_omnia_baseline import hog_intensity


def test_hog_intensity_is_fixed_finite_and_spatially_sensitive():
    images = np.zeros((2, 32, 32), dtype=np.float32)
    images[0, :, 8:12] = 1
    images[1, 8:12, :] = 1
    first = hog_intensity(images)
    second = hog_intensity(images.copy())
    assert first.shape == (2, 580)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])


def test_hog_intensity_requires_32px_batch():
    with pytest.raises(ValueError):
        # NumPy reshape is the input contract enforcement in this audit tool.
        hog_intensity(np.zeros((1, 16, 16), dtype=np.float32))
