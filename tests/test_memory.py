import json
import numpy as np
import pytest
from mycelia import AssociativeMycelium,MemoryColony,MemoryConfig,image_cues


def pattern(axis=0):
    x=np.zeros((5,5));x[1:4,1:4]=.8
    if axis:x=np.rot90(np.triu(np.ones((5,5)))*.8)
    return image_cues(x)


def test_imprint_persists_inside_compartments_and_reinforces_real_cords():
    c=MemoryColony(MemoryConfig(size=5));cue=pattern()
    radius=[e.radius for e in c.organism.segments.values()]
    result=c.expose(cue)
    assert result['changed_nodes']>0
    assert all(n.receptor_trace for n in c.organism.nodes.values())
    assert sum(n.memory_material for n in c.organism.nodes.values())>0
    assert any(e.radius>r for e,r in zip(c.organism.segments.values(),radius))
    assert c.distance(cue)<c.distance(pattern(1))
    c.organism._validate()


def test_training_changes_recognition_without_external_readout():
    m=AssociativeMycelium(['square','triangle'],MemoryConfig(size=5,colonies_per_label=1))
    assert m.predict(pattern()) is None
    m.learn(pattern(),'square');m.learn(pattern(1),'triangle')
    assert m.predict(pattern())=='square'
    assert m.predict(pattern(1))=='triangle'


def test_frozen_recall_does_not_mutate_graph_rng_or_pools():
    m=AssociativeMycelium(['square'],MemoryConfig(size=5));m.learn(pattern(),'square');m.freeze()
    before=json.dumps(m.state_dict(),sort_keys=True)
    for _ in range(3):assert m.predict(pattern())=='square'
    assert json.dumps(m.state_dict(),sort_keys=True)==before
    with pytest.raises(RuntimeError):m.learn(pattern(),'square')
    assert json.dumps(m.state_dict(),sort_keys=True)==before


def test_zero_resource_blocks_synthesis_and_zero_reward_blocks_memory():
    c=MemoryColony(MemoryConfig(size=5));before=c.state_dict()
    c.expose(pattern(),reward=0);assert c.state_dict()==before
    for n in c.organism.nodes.values():
        n.reserve+=n.nutrient;n.nutrient=0
        c.organism.energy_dissipated+=n.energy;n.energy=0
    result=c.expose(pattern(),feed=0)
    assert result['synthesis_carbon']==0
    assert all(not n.receptor_trace for n in c.organism.nodes.values())
    c.organism._validate()


def test_turnover_forgets_and_recycles_memory_material_conservatively():
    c=MemoryColony(MemoryConfig(size=5));cue=pattern()
    for _ in range(5):c.expose(cue)
    before=c.distance(cue);carbon=c.organism.total_carbon()
    c.rest(500)
    assert c.distance(cue)>before
    assert c.organism.total_carbon()==pytest.approx(carbon)
    c.organism._validate()


def test_serialized_memory_resumes_and_recall_is_identical(tmp_path):
    m=AssociativeMycelium(['square','triangle'],MemoryConfig(size=5,colonies_per_label=1,physiology_interval=2))
    m.learn(pattern(),'square');m.learn(pattern(1),'triangle')
    p=tmp_path/'memory.json';m.save(p);restored=AssociativeMycelium.load(p)
    assert restored.scores(pattern())==m.scores(pattern())
    m.learn(pattern(),'square');restored.learn(pattern(),'square')
    assert restored.state_dict()==m.state_dict()


def test_repeated_exposures_and_physiology_preserve_all_budgets():
    c=MemoryColony(MemoryConfig(size=5,physiology_interval=2))
    for i in range(24):c.expose(pattern(i%2))
    s=c.organism.summary()
    assert all(abs(s[k])<1e-8 for k in ('carbon_error','water_error','energy_error'))


def test_invalid_cue_is_rejected_before_modification():
    c=MemoryColony(MemoryConfig(size=5));before=c.state_dict()
    with pytest.raises(ValueError):c.expose(np.full((5,5,3),np.nan))
    assert c.state_dict()==before


def test_failed_exposure_rolls_back_synthesis_environment_and_energy(monkeypatch):
    c=MemoryColony(MemoryConfig(size=5,physiology_interval=1));before=c.state_dict();env=c.organism.environment
    def fail():raise ArithmeticError('injected physiology failure')
    monkeypatch.setattr(c.organism,'step',fail)
    with pytest.raises(ArithmeticError):c.expose(pattern())
    assert c.state_dict()==before
    assert c.organism.environment is env


def test_old_organism_states_load_with_empty_memory():
    from mycelia import Mycelium,Environment
    original=Mycelium(Environment(np.ones((5,5))))
    data=original.state_dict();data.pop('memory_frozen');data['scalars'].pop('memory_spent')
    for n in data['nodes']:
        for key in ('receptor_trace','receptor_variance','memory_material','memory_exposures'):n.pop(key)
    restored=Mycelium.from_state_dict(data)
    assert restored.memory_spent==0
    assert all(not n.receptor_trace for n in restored.nodes.values())
    restored.step();restored._validate()


def test_failed_first_lesson_leaves_no_phantom_colony(monkeypatch):
    m=AssociativeMycelium(['square'],MemoryConfig(size=5))
    before=m.state_dict()
    def fail(self,*args,**kwargs):raise ArithmeticError('injected failure')
    monkeypatch.setattr(MemoryColony,'expose',fail)
    with pytest.raises(ArithmeticError):m.learn(pattern(),'square')
    assert m.state_dict()==before
    assert m.predict(pattern()) is None


def test_cords_require_energy_and_account_for_it():
    c=MemoryColony(MemoryConfig(size=5))
    result=c.expose(pattern())
    assert result['synthesis_energy']==pytest.approx(result['synthesis_carbon']*c.config.energy_cost)
    for n in c.organism.nodes.values():
        c.organism.energy_dissipated+=n.energy;n.energy=0
    before=[(e.material,e.radius,e.route_trace) for e in c.organism.segments.values()]
    result=c.expose(pattern(1),feed=0)
    assert result['synthesis_carbon']==0
    assert [(e.material,e.radius,e.route_trace) for e in c.organism.segments.values()]==before
    c.organism._validate()


def test_batch_scores_match_scalar_and_read_actual_state():
    m=AssociativeMycelium(['square','triangle'],MemoryConfig(size=5,colonies_per_label=2))
    cues=np.stack([pattern(),pattern(1)])
    m.learn(cues[0],'square');m.learn(cues[1],'triangle');m.freeze()
    before=json.dumps(m.state_dict(),sort_keys=True)
    expected=np.array([list(m.scores(c).values()) for c in cues])
    np.testing.assert_allclose(m.score_many(cues,batch_size=1),expected,atol=1e-14)
    assert m.predict_many(cues)==[m.predict(c) for c in cues]
    assert json.dumps(m.state_dict(),sort_keys=True)==before
    # Reordering a loaded dictionary must not change spatial image registration.
    for pop in m.colonies.values():
        for colony in pop:
            colony.organism.nodes=dict(reversed(list(colony.organism.nodes.items())))
    np.testing.assert_allclose(m.score_many(cues),expected,atol=1e-14)
    np.testing.assert_allclose([list(m.scores(c).values()) for c in cues],expected,atol=1e-14)
    for pop in m.colonies.values():
        for colony in pop:
            for n in colony.organism.nodes.values():n.memory_material=0
    assert m.predict_many(cues)==[None,None]
    assert m.predict(cues[0]) is None


def test_batch_validation_even_without_learned_colonies():
    m=AssociativeMycelium(['square'],MemoryConfig(size=5))
    assert m.score_many(np.empty((0,5,5,3))).shape==(0,1)
    assert m.predict_many(np.stack([pattern()]))==[None]
    for invalid in (np.zeros((5,5,3)),np.full((1,5,5,3),np.nan)):
        with pytest.raises(ValueError):m.score_many(invalid)
    with pytest.raises(ValueError):m.scores(np.full((5,5,3),np.nan))
    with pytest.raises(ValueError):m.score_many(np.stack([pattern()]),batch_size=True)


def test_rest_failure_is_transactional(monkeypatch):
    c=MemoryColony(MemoryConfig(size=5));c.expose(pattern())
    before=c.state_dict();env=c.organism.environment
    def fail():raise ArithmeticError('injected validation failure')
    monkeypatch.setattr(c.organism,'_validate',fail)
    with pytest.raises(ArithmeticError):c.rest(100)
    assert c.state_dict()==before
    assert c.organism.environment is env


def test_loading_rejects_inconsistent_memory_checkpoint(tmp_path):
    from copy import deepcopy
    m=AssociativeMycelium(['square'],MemoryConfig(size=5));m.learn(pattern(),'square');m.freeze()
    original=m.state_dict();p=tmp_path/'bad.json'
    for field,value in [('frozen','false'),('exposures',1.5),('consolidation','false'),('energy_cost',.8)]:
        data=deepcopy(original)
        if field=='frozen':data[field]=value
        elif field=='exposures':data['colonies']['square'][0][field]=value
        else:data['colonies']['square'][0]['config'][field]=value
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError):AssociativeMycelium.load(p)
    data=deepcopy(original);data['colonies']['square'][0]['organism']['memory_frozen']=False
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):AssociativeMycelium.load(p)
    # Version 1 checkpoints omitted consolidation; False is the compatible default.
    data=deepcopy(original);data['config'].pop('consolidation')
    data['colonies']['square'][0]['config'].pop('consolidation')
    p.write_text(json.dumps(data));restored=AssociativeMycelium.load(p)
    assert restored.predict(pattern())=='square'


def test_equal_score_ties_survive_reordered_checkpoint_populations(tmp_path):
    m=AssociativeMycelium(['first','second'],MemoryConfig(size=5,colonies_per_label=1))
    # Identical imprints deliberately create the boundary case for tie breaking.
    m.learn(pattern(),'first');m.learn(pattern(),'second');m.freeze()
    data=m.state_dict();data['colonies']=dict(reversed(list(data['colonies'].items())))
    path=tmp_path/'reordered.json';path.write_text(json.dumps(data))
    restored=AssociativeMycelium.load(path)
    before=json.dumps(restored.state_dict(),sort_keys=True)
    assert list(restored.scores(pattern()))==['first','second']
    assert restored.predict(pattern())=='first'
    assert restored.predict_many(np.stack([pattern()]))==['first']
    assert json.dumps(restored.state_dict(),sort_keys=True)==before
