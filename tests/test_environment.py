import numpy as np
import pytest

from mycelia import Environment


def test_uniform_rich_field_is_preserved():
    e = Environment(np.full((8,8),2.0))
    e.step(20)
    assert np.allclose(e.nutrient,2.0)


def test_diffusion_conserves_and_smooths_with_large_timestep():
    a = np.zeros((12,12)); a[6,6]=1
    e = Environment(a)
    e.step(30)
    assert e.nutrient.sum()==pytest.approx(1)
    assert e.nutrient.min()>=0
    assert 0<e.nutrient[6,6]<1


def test_uptake_reports_actual_available_amount():
    e = Environment(np.ones((3,3)))
    assert e.consume(1,1,10)==1
    assert e.consume(1,1,10)==0
    assert e.nutrient.sum()==8


def test_no_gradient_is_zero():
    e = Environment(np.ones((8,8)))
    assert e.gradient(4,4)==(0,0)


@pytest.mark.parametrize('value',[-1,float('nan'),float('inf')])
def test_invalid_resource_is_rejected(value):
    with pytest.raises(ValueError):
        Environment(np.full((3,3),value))

