import json
import numpy as np
import pytest

from mycelia import Config, Environment, Mycelium
from mycelia.environment import scenario


def organism(name='uniform',**config):
    c=Config(**config)
    return Mycelium(scenario(name,32,c.diffusion),c,seed=7)


def test_integrated_run_grows_branches_transports_and_conserves():
    o=organism(branch_interval=3)
    s=o.run(120)
    assert s['nodes']>10
    assert s['events'].get('branch',0)>0
    assert s['total_flow']>0
    assert s['uptake']>0
    assert abs(s['carbon_error'])<1e-8
    assert abs(s['water_error'])<1e-8
    assert abs(s['energy_error'])<1e-8
    assert all(0<=e.pore_open<=1 for e in o.segments.values())


def test_rich_environment_supports_more_growth_than_poor():
    rich,poor=organism(),organism('poor')
    rich.run(80);poor.run(80)
    assert rich.summary()['total_length']>poor.summary()['total_length']
    assert rich.uptake>poor.uptake


def test_position_is_honored():
    o=Mycelium(Environment(np.ones((10,10))),position=(2,3))
    assert (o.nodes[0].x,o.nodes[0].y)==(2,3)


def test_duplicate_fusion_is_not_counted_twice():
    o=organism()
    n=o._node(17,16,1,water=1,nutrient=.1,energy=.1,osmolyte=0,reserve=0,biomass=.1)
    _,first=o.connect(0,n.id,fusion=True)
    _,second=o.connect(n.id,0,fusion=True)
    assert first and not second
    assert o.counts['fusion']==1


def test_state_roundtrip_resumes_identically(tmp_path):
    o=organism(branch_interval=3);o.run(20)
    p=tmp_path/'state.json';o.save(p)
    resumed=Mycelium.load(p)
    o.run(15);resumed.run(15)
    assert o.state_dict()==resumed.state_dict()


def test_failure_rolls_back_all_state_and_environment(monkeypatch):
    o=organism();before=json.dumps(o.state_dict(),sort_keys=True)
    env=o.environment
    def failed(dt):
        o.nodes[0].nutrient+=5
        o.environment.nutrient[0,0]=0
        raise ArithmeticError('injected failure')
    monkeypatch.setattr(o,'_growth',failed)
    with pytest.raises(ArithmeticError):o.step()
    assert o.environment is env
    assert json.dumps(o.state_dict(),sort_keys=True)==before


def test_resource_interventions_preserve_explicit_budget():
    o=organism();o.add_resource_patch(10,10,3,2);o.deplete_region(20,20,2)
    s=o.run(10)
    assert abs(s['carbon_error'])<1e-8


def test_node_and_history_limits():
    o=organism(max_nodes=25,max_tips=8,event_limit=30,branch_interval=2)
    o.run(80)
    assert len(o.nodes)<=25
    assert sum(t.active for t in o.tips.values())<=8
    assert len(o.events)<=30


@pytest.mark.parametrize('dt',[0,-1,float('nan'),float('inf')])
def test_invalid_timestep_does_not_mutate(dt):
    o=organism();before=o.state_dict()
    with pytest.raises(ValueError):o.step(dt)
    assert before==o.state_dict()


def test_large_steps_match_same_internal_substeps():
    a,b=organism(),organism()
    a.run(20)
    for _ in range(10):b.step(1)
    assert a.summary()['total_length']==pytest.approx(b.summary()['total_length'])
    assert a.summary()['total_flow']==pytest.approx(b.summary()['total_flow'])


def test_obstacle_cells_are_never_entered():
    o=organism('obstacle');o.run(80)
    assert all(not o.environment.blocked[o.environment.cell(n.x,n.y)] for n in o.nodes.values())
    assert all(o.environment.path_clear(o.nodes[e.a].x,o.nodes[e.a].y,o.nodes[e.b].x,o.nodes[e.b].y) for e in o.segments.values())


def test_contact_fusion_closes_a_persistent_loop():
    o=organism();root=o.nodes[0]
    previous=0
    for x,y in ((root.x+2,root.y),(root.x+2,root.y+2),(root.x,root.y+2),(root.x+.3,root.y)):
        n=o._node(x,y,1,water=1,nutrient=.1,energy=.1,osmolyte=0,reserve=0,biomass=.1)
        o.connect(previous,n.id)
        previous=n.id
    tip=o._tip(previous,1,0);tip.branch_age=3
    o._fusion();o._fusion()
    assert o.counts['fusion']==1
    assert len(o.segments)==len(o.nodes)  # one connected cycle


def test_dormant_tip_reactivates_when_surviving_tissue_is_refueled():
    o=organism(resprout_threshold=.03)
    for tip in o.tips.values():tip.active=False
    root=o.nodes[0]
    o.add_resource_patch(root.x,root.y,radius=2,strength=2)
    s=o.step()
    assert o.counts['tip_reactivation']>0
    assert sum(t.active for t in o.tips.values())>0
    assert abs(s['carbon_error'])<1e-8
    assert abs(s['water_error'])<1e-8
    assert abs(s['energy_error'])<1e-8


def test_living_viable_compartment_can_resprout_without_a_saved_tip():
    o=organism(resprout_threshold=.03)
    o.tips.clear()
    root=o.nodes[0]
    o.add_resource_patch(root.x,root.y,radius=2,strength=2)
    s=o.step()
    assert o.counts['tip_resprout']>0
    assert sum(t.active for t in o.tips.values())>0
    assert all(t.node in o.nodes for t in o.tips.values())
    assert abs(s['carbon_error'])<1e-8
    assert abs(s['water_error'])<1e-8
    assert abs(s['energy_error'])<1e-8


def test_zero_gradient_retains_tip_polarity():
    o=organism();tip=next(iter(o.tips.values()));tip.dx,tip.dy=0,1
    o._direction(tip,o.nodes[tip.node],.5)
    assert (tip.dx,tip.dy)==(0,1)


def test_damage_closes_septa_and_reduces_flow():
    o=organism();o.run(10)
    for n in o.nodes.values():n.damage=1
    o.environment.stress[:]=2
    o.step()
    assert all(e.pore_open==0 for e in o.segments.values() if e.alive)
    assert o.summary()['total_flow']==0


def test_terminal_retraction_returns_carbon_and_water():
    o=organism(starvation_time=1);s=o.run(15)
    prior_retractions=o.counts['retraction']
    for tip in o.tips.values():tip.starvation=o.config.starvation_time+1
    for n in o.nodes.values():
        # Convert existing nutrients to an explicit respiration sink rather
        # than delete them while preparing the starvation intervention.
        o.respired+=n.nutrient
        n.energy+=n.nutrient*o.config.energy_yield
        n.nutrient=0
    for e in o.segments.values():e.activity=0;e.inactive_time=10
    o._remodel(.5)
    assert o.counts['retraction']>0
    assert len(o.nodes)==s['nodes']-(o.counts['retraction']-prior_retractions)
    o._validate()


def test_retracted_storage_capacity_can_be_used_for_later_growth(tmp_path):
    config=Config(max_nodes=8,max_tips=8,branch_interval=2,starvation_time=2)
    o=Mycelium(Environment(np.zeros((32,32))),config,seed=7)
    o.run(80)
    assert o.counts['retraction']>0
    assert len(o.nodes)<config.max_nodes
    created_before=o.next_node
    checkpoint=tmp_path/'recycled-state.json'
    o.save(checkpoint)
    resumed=Mycelium.load(checkpoint)
    assert resumed.state_dict()==o.state_dict()

    # A later resource pulse restarts growth from a new inoculated front. The
    # newly created IDs prove released storage is reused after save/reload too.
    for network in (o,resumed):
        root=network.nodes[0]
        network.add_resource_patch(root.x,root.y,5,50)
        for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
            network._tip(root.id,dx,dy)
        network.run(120)

    assert o.next_node>created_before
    assert len(o.nodes)<=config.max_nodes
    assert o.counts['extension']>3
    assert abs(o.summary()['carbon_error'])<1e-8
    assert abs(o.summary()['water_error'])<1e-8
    assert abs(o.summary()['energy_error'])<1e-8
    assert o.state_dict()==resumed.state_dict()
    o._validate()


def test_large_timestep_preserves_budgets():
    o=organism(max_nodes=100);s=o.step(10)
    assert abs(s['carbon_error'])<1e-8
    assert abs(s['water_error'])<1e-8
    assert abs(s['energy_error'])<1e-8
