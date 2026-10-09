"""Resource gates, aging, saturation, turnover and restart for candidate plasticity."""
from copy import deepcopy
import json
import numpy as np
import pytest
from mycelia import MemoryConfig,MemoryColony,AssociativeMycelium,image_cues


def cues():return image_cues(np.eye(5)),image_cues(np.fliplr(np.eye(5)))


def cap_cords(colony):
    o=colony.organism
    for e in o.segments.values():
        cost=2*(o.config.radius-e.radius);a,b=o.nodes[e.a],o.nodes[e.b]
        assert min(a.nutrient,b.nutrient)>=cost/2
        a.nutrient-=cost/2;b.nutrient-=cost/2;e.material+=cost;e.radius=o.config.radius
        energy=cost*colony.config.energy_cost
        a.energy-=energy/2;b.energy-=energy/2;o.memory_spent+=energy
    o._validate()


def test_adapts_to_new_cue_while_retaining_slow_old_trace():
    a,b=cues();c=MemoryColony(MemoryConfig(size=5,adaptive=True))
    for _ in range(40):c.expose(a)
    initial=c.distance(b)
    for _ in range(12):c.expose(b)
    assert c.distance(b)<initial*.01
    assert c.distance(a)<initial*.05
    for _ in range(5):c.expose(a)
    assert c.distance(a)<initial*.01
    c.organism._validate()


def test_aging_does_not_disable_fast_plasticity():
    a,b=cues();c=MemoryColony(MemoryConfig(size=5,adaptive=True,consolidation=True))
    for _ in range(180):c.expose(a)
    before=c.distance(b)
    for _ in range(6):c.expose(b)
    assert c.distance(b)<before*.01


def test_cord_chemistry_updates_after_physical_radius_saturates():
    a,b=cues();c=MemoryColony(MemoryConfig(size=5,adaptive=True));c.expose(a);cap_cords(c)
    old=[(e.radius,e.material,e.route_fast_trace) for e in c.organism.segments.values()]
    update=c.expose(b)
    assert any(e.route_fast_trace!=v[2] for e,v in zip(c.organism.segments.values(),old))
    assert all((e.radius,e.material)==v[:2] for e,v in zip(c.organism.segments.values(),old))
    assert sum(e.route_fast_material+e.route_memory_material for e in c.organism.segments.values())>0
    assert update['synthesis_energy']==pytest.approx(update['synthesis_carbon']*c.config.energy_cost)
    c.organism._validate()


@pytest.mark.parametrize('initially_trained',[False,True])
def test_zero_energy_blocks_both_channels_and_cord_turnover(initially_trained):
    a,b=cues();c=MemoryColony(MemoryConfig(size=5,adaptive=True))
    if initially_trained:c.expose(a)
    for n in c.organism.nodes.values():c.organism.energy_dissipated+=n.energy;n.energy=0
    traces=[(n.receptor_trace,n.receptor_fast_trace) for n in c.organism.nodes.values()]
    cords=[(e.route_trace,e.route_fast_trace,e.material,e.route_memory_material,e.route_fast_material) for e in c.organism.segments.values()]
    update=c.expose(b,feed=0)
    assert update['synthesis_carbon']==update['synthesis_energy']==0
    assert traces==[(n.receptor_trace,n.receptor_fast_trace) for n in c.organism.nodes.values()]
    assert cords==[(e.route_trace,e.route_fast_trace,e.material,e.route_memory_material,e.route_fast_material) for e in c.organism.segments.values()]
    c.organism._validate()


def test_zero_reward_is_noop_in_adaptive_mode():
    a,_=cues();c=MemoryColony(MemoryConfig(size=5,adaptive=True));before=c.state_dict()
    c.expose(a,reward=0)
    assert before==c.state_dict()


def test_turnover_recycles_both_pools_at_different_rates():
    a,_=cues();c=MemoryColony(MemoryConfig(size=5,adaptive=True));c.expose(a)
    carbon=c.organism.total_carbon();fast=sum(n.fast_memory_material for n in c.organism.nodes.values());slow=sum(n.memory_material for n in c.organism.nodes.values())
    c.rest(20)
    assert sum(n.fast_memory_material for n in c.organism.nodes.values())/fast<sum(n.memory_material for n in c.organism.nodes.values())/slow
    assert c.organism.total_carbon()==pytest.approx(carbon)
    c.rest(1e8)
    assert c.distance(a)==float('inf')
    c.organism._validate()


def test_failed_adaptive_step_rolls_back_every_new_pool(monkeypatch):
    a,_=cues();c=MemoryColony(MemoryConfig(size=5,adaptive=True,physiology_interval=1));before=c.state_dict();env=c.organism.environment
    def fail():raise ArithmeticError('injected failure')
    monkeypatch.setattr(c.organism,'step',fail)
    with pytest.raises(ArithmeticError):c.expose(a)
    assert c.state_dict()==before and c.organism.environment is env


@pytest.mark.parametrize('fast,slow',[(.45,.015),(.45,0),(0,.015)])
def test_adaptive_batch_save_load_resume_and_ablation(fast,slow,tmp_path):
    a,b=cues();m=AssociativeMycelium(['a','b'],MemoryConfig(size=5,adaptive=True,fast_rate=fast,slow_rate=slow,colonies_per_label=1,physiology_interval=2))
    for _ in range(4):m.learn(a,'a');m.learn(b,'b')
    path=tmp_path/'memory.json';m.save(path);restored=AssociativeMycelium.load(path)
    m.learn(b,'a');restored.learn(b,'a');assert m.state_dict()==restored.state_dict()
    m.freeze();before=json.dumps(m.state_dict(),sort_keys=True)
    batch=m.score_many(np.stack([a,b]),batch_size=1)
    np.testing.assert_allclose(batch,np.array([list(m.scores(c).values()) for c in (a,b)]),atol=1e-14)
    assert json.dumps(m.state_dict(),sort_keys=True)==before
    for pop in m.colonies.values():
        for colony in pop:
            for n in colony.organism.nodes.values():
                n.receptor_trace=[];n.receptor_variance=[];n.receptor_fast_trace=[]
    assert m.predict_many(np.stack([a,b]))==[None,None]


def test_adaptive_checkpoint_rejects_corrupt_new_state(tmp_path):
    a,_=cues();m=AssociativeMycelium(['a'],MemoryConfig(size=5,adaptive=True));m.learn(a,'a');state=m.state_dict()
    for key,value in [('fast_memory_material',float('nan')),('fast_memory_exposures',-1),('receptor_fast_trace',[2,0,0])]:
        data=deepcopy(state);data['colonies']['a'][0]['organism']['nodes'][0][key]=value
        path=tmp_path/'bad.json';path.write_text(json.dumps(data))
        with pytest.raises((ValueError,ArithmeticError)):AssociativeMycelium.load(path)


@pytest.mark.parametrize('kwargs',[{'adaptive':'false'},{'adaptive':True,'fast_rate':0,'slow_rate':0},{'fast_rate':1.5},{'route_memory_cost':0},{'stability_scale':0}])
def test_invalid_adaptive_configuration(kwargs):
    with pytest.raises(ValueError):MemoryConfig(**kwargs)
