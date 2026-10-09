from copy import deepcopy
import numpy as np
import pytest
from mycelia import Mycelium,Environment,MemoryColony,MemoryConfig,AssociativeMycelium,image_cues


@pytest.mark.parametrize('field',['initial_carbon','initial_water','initial_energy','respired','memory_spent','time'])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,True])
def test_bad_checkpoint_ledgers_are_rejected(field,value):
    state=Mycelium(Environment(np.ones((5,5)))).state_dict();state['scalars'][field]=value
    with pytest.raises(ValueError):Mycelium.from_state_dict(state)


@pytest.mark.parametrize('collection',['nodes','segments','tips'])
def test_duplicate_checkpoint_ids_are_rejected(collection):
    o=Mycelium(Environment(np.ones((5,5))))
    if collection=='segments':
        n=o._node(1,1,1,water=0,nutrient=0,energy=0,osmolyte=0,reserve=0,biomass=0);o.connect(0,n.id)
        o.initial_carbon=o.total_carbon()
    state=o.state_dict();state[collection].append(deepcopy(state[collection][0]))
    with pytest.raises(ValueError):Mycelium.from_state_dict(state)


@pytest.mark.parametrize('bad',['false',0,None])
def test_malformed_freeze_flags_are_rejected(bad):
    state=Mycelium(Environment(np.ones((5,5)))).state_dict();state['memory_frozen']=bad
    with pytest.raises(ValueError):Mycelium.from_state_dict(state)


def test_checkpoint_masks_counters_and_scalar_names_are_validated():
    state=Mycelium(Environment(np.ones((5,5)))).state_dict()
    for change in ('mask_shape','mask_type','allocation_counter','extra_scalar'):
        data=deepcopy(state)
        if change=='mask_shape':data['environment']['blocked']=[[False]]
        elif change=='mask_type':data['environment']['blocked']=[[0]*5 for _ in range(5)]
        elif change=='allocation_counter':data['scalars']['next_node']=0
        else:data['scalars']['environment']={}
        with pytest.raises(ValueError):Mycelium.from_state_dict(data)


def test_retired_tissue_cannot_learn_or_recognize():
    m=AssociativeMycelium(['pattern'],MemoryConfig(size=3));cue=image_cues(np.eye(3));m.learn(cue,'pattern')
    c=m.colonies['pattern'][0]
    for e in c.organism.segments.values():e.alive=False
    for n in c.organism.nodes.values():n.alive=False
    c.organism._validate();before=c.state_dict()
    assert c.expose(cue)['synthesis_carbon']==0
    assert c.state_dict()==before
    assert m.predict(cue) is None
    assert m.predict_many(cue[None])==[None]


def test_scalar_batch_agree_after_partial_tissue_retirement():
    m=AssociativeMycelium(['pattern'],MemoryConfig(size=3));cue=image_cues(np.eye(3));m.learn(cue,'pattern')
    c=m.colonies['pattern'][0];c.organism.nodes[0].alive=False
    for e in c.organism.segments.values():
        if e.a==0 or e.b==0:e.alive=False
    c.organism._validate()
    assert m.score_many(cue[None])[0,0]==pytest.approx(m.scores(cue)['pattern'],abs=1e-14)
