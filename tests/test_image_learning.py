import importlib.util
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('sklearn')
pytest.importorskip('PIL')
spec=importlib.util.spec_from_file_location('accuracy_test',Path(__file__).parents[1]/'tools/accuracy_test.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_constant_image_has_zero_gradient_and_retains_intensity():
    x=module.image_features(np.full((64,64),.25))
    assert x.shape==(2020,)
    assert np.all(x[:1764]==0)
    assert np.all(x[1764:]==.25)

def test_gradient_features_are_finite_and_respond_to_orientation():
    x=np.tile(np.linspace(0,1,64),(64,1))
    horizontal=module.image_features(x);vertical=module.image_features(x.T)
    assert np.all(np.isfinite(horizontal))
    assert not np.allclose(horizontal[:1764],vertical[:1764])
