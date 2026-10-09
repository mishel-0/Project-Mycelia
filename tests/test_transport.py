import numpy as np
import pytest

from mycelia import Config, Environment, Mycelium
from mycelia.transport import pressure_solve, transfer, transport


def pair():
    o = Mycelium(Environment(np.ones((12,12))),Config(streaming_rate=0,solute_diffusion=0),position=(3,3))
    a = o.nodes[0]
    b = o._node(4,3,1,water=a.volume*.8,nutrient=.1,energy=.2,osmolyte=.1,reserve=0,biomass=.1)
    a.water=a.volume*1.4
    a.pressure=(a.water/a.volume-1)/o.config.wall_compliance
    b.pressure=(b.water/b.volume-1)/o.config.wall_compliance
    o.connect(a.id,b.id)
    return o


def test_water_flow_follows_pressure_and_conserves_volume():
    o = pair(); before=sum(n.water for n in o.nodes.values())
    transport(o,.5)
    assert o.segments[0].flow>0
    assert sum(n.water for n in o.nodes.values())==pytest.approx(before)
    assert o.pressure_residual<1e-9


def test_closed_septum_blocks_water_and_solute():
    o=pair(); o.segments[0].pore_open=0
    before=[(n.water,n.nutrient) for n in o.nodes.values()]
    transport(o,10)
    assert [(n.water,n.nutrient) for n in o.nodes.values()]==before
    assert o.segments[0].flow==0


def test_simultaneous_outgoing_transfers_share_donor_budget():
    o=pair()
    b=o._node(5,3,1,water=1,nutrient=0,energy=0,osmolyte=0,reserve=0,biomass=0)
    o.nodes[0].nutrient=1
    initial=sum(n.nutrient for n in o.nodes.values())
    transfer(o,'nutrient',[(0,0,1,10),(1,0,b.id,10)])
    assert o.nodes[0].nutrient==0
    assert o.nodes[1].nutrient==pytest.approx(.6)
    assert b.nutrient==pytest.approx(.5)
    assert sum(n.nutrient for n in o.nodes.values())==pytest.approx(initial)


def test_disconnected_pressure_system_has_no_artificial_sources():
    o=pair(); o.segments[0].alive=False
    p,error=pressure_solve(o,.5)
    assert p[0]==pytest.approx(o.nodes[0].pressure)
    assert p[1]==pytest.approx(o.nodes[1].pressure)
    assert error<1e-10


def test_backward_flow_preserves_sign_and_activity():
    o=pair(); a,b=o.nodes.values(); a.water,b.water=b.water,a.water
    transport(o,.5)
    assert o.segments[0].flow<0
    assert o.segments[0].activity>0

