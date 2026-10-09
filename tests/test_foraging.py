import json,math
import numpy as np
import pytest
from mycelia import Config,Environment,Mycelium
from mycelia.foraging import transfer_inoculum,direction_metrics


def exposed():
    yy,xx=np.indices((16,16));c=Config(growth_speed=0,branch_rate=0,foraging_memory_rate=.3,foraging_memory_gain=1.2)
    o=Mycelium(Environment(.2+.02*xx),c,position=(8,8));o.run(20)
    return o


def test_local_uptake_and_gradient_create_paid_directional_state():
    o=exposed();n=o.nodes[0]
    assert n.polarity_trace_x>0 and abs(n.polarity_trace_y)<1e-10
    assert n.polarity_material>0 and o.memory_spent>0
    s=o.summary()
    assert max(abs(s[k]) for k in ('carbon_error','water_error','energy_error'))<1e-8


def test_gradient_without_uptake_does_not_consolidate_memory():
    _,xx=np.indices((16,16));c=Config(uptake_rate=0,growth_speed=0,foraging_memory_rate=.3)
    o=Mycelium(Environment(.2+.02*xx),c,position=(8,8));o.run(20)
    assert o.nodes[0].polarity_trace_x==0
    assert o.nodes[0].polarity_material==0


def test_uniform_field_without_growth_has_no_directional_imprint():
    o=Mycelium(Environment(np.ones((16,16))),Config(growth_speed=0,foraging_memory_rate=.3),position=(8,8));o.run(20)
    assert math.hypot(o.nodes[0].polarity_trace_x,o.nodes[0].polarity_trace_y)<1e-12


def test_inoculum_transfer_removes_old_geometry_medium_and_polarity():
    source=exposed();before=json.dumps(source.state_dict(),sort_keys=True)
    a=transfer_inoculum(source,seed=9);b=transfer_inoculum(source,seed=9,condition='erased')
    assert json.dumps(source.state_dict(),sort_keys=True)==before
    assert len(a.nodes)==1 and not a.segments
    assert np.all(a.environment.nutrient==.6)
    assert a.environment is not source.environment
    assert a.config.foraging_memory_rate==0
    assert [(t.dx,t.dy) for t in a.tips.values()]==[(t.dx,t.dy) for t in b.tips.values()]
    assert b.nodes[0].polarity_trace_x==0 and a.nodes[0].polarity_trace_x>0
    assert a.total_carbon()==b.total_carbon()
    assert a.total_water()==b.total_water()
    assert a.initial_energy==b.initial_energy
    a._validate();b._validate()


def test_rotated_trace_rotates_response_and_gain_off_removes_bias():
    source=exposed();a=transfer_inoculum(source);r=transfer_inoculum(source,condition='rotated');off=transfer_inoculum(source,condition='bias_off')
    for o in (a,r,off):
        tip=next(iter(o.tips.values()));tip.dx,tip.dy=0,1
        o._direction(tip,o.nodes[0],.5)
    assert next(iter(a.tips.values())).dx>0
    assert next(iter(r.tips.values())).dx<0
    assert next(iter(off.tips.values())).dx==0


def test_directional_state_serializes_and_resumes_identically(tmp_path):
    o=exposed();p=tmp_path/'state.json';o.save(p);r=Mycelium.load(p)
    o.run(4);r.run(4)
    assert o.state_dict()==r.state_dict()


def test_decay_recycles_polarity_carbon():
    o=exposed();n=o.nodes[0];before=n.polarity_material;carbon=o.total_carbon();strength=n.polarity_trace_x
    o.config=Config(growth_speed=0,foraging_memory_decay=.2)
    o._polarity_memory(n,0,1)
    assert n.polarity_material<before and n.polarity_trace_x<strength
    assert o.total_carbon()==pytest.approx(carbon)
    o._validate()


def test_no_growth_has_no_direction_metric():
    assert direction_metrics(exposed(),0)['alignment'] is None


def test_default_build_does_not_enable_new_memory():
    o=Mycelium(Environment(np.ones((16,16))));o.run(20)
    assert all(n.polarity_material==0 for n in o.nodes.values())


def test_active_memory_cannot_have_free_synthesis():
    with pytest.raises(ValueError):Config(foraging_memory_rate=.1,foraging_memory_cost=0)
    with pytest.raises(ValueError):Config(foraging_memory_inheritance=1.1)


def test_new_growth_inherits_and_dilutes_paid_memory_without_creating_carbon():
    from dataclasses import replace
    source=exposed();source.config=replace(source.config,growth_speed=.65)
    o=transfer_inoculum(source);o.run(70)
    descendants=[n for n in o.nodes.values() if n.id!=0]
    assert descendants
    assert any(n.polarity_material>0 and n.polarity_trace_x>0 for n in descendants)
    assert max(n.polarity_trace_x for n in descendants)<o.nodes[0].polarity_trace_x
    o._validate()


def test_failure_rolls_back_directional_synthesis(monkeypatch):
    o=exposed();before=o.state_dict()
    def fail(dt):raise ArithmeticError('injected growth failure')
    monkeypatch.setattr(o,'_growth',fail)
    with pytest.raises(ArithmeticError):o.step()
    assert o.state_dict()==before


def test_old_state_without_polarity_fields_loads():
    o=Mycelium(Environment(np.ones((8,8))));data=o.state_dict()
    for name in tuple(data['config']):
        if name.startswith('foraging_memory_'):data['config'].pop(name)
    for n in data['nodes']:
        for name in ('polarity_trace_x','polarity_trace_y','polarity_material'):n.pop(name)
    restored=Mycelium.from_state_dict(data)
    assert restored.nodes[0].polarity_material==0
    restored.step();restored._validate()
