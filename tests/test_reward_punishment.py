import numpy as np, pytest
from mycelia import AssociativeMycelium, MemoryColony, MemoryConfig, image_cues


def cue(seed,size=8):
    return image_cues(np.random.default_rng(seed).random((size,size)))


def test_punish_moves_trace_away_and_keeps_budgets():
    colony=MemoryColony(MemoryConfig(size=8));x=cue(1)
    for _ in range(3):colony.expose(x)
    before=colony.distance(x)
    result=colony.punish(x,penalty=1.)
    assert result['changed_nodes']>0 and result['synthesis_carbon']>0
    assert colony.distance(x)>before
    s=colony.organism.summary()
    assert all(abs(s[k])<1e-8 for k in ('carbon_error','water_error','energy_error'))


def test_punish_rejects_invalid_and_frozen():
    colony=MemoryColony(MemoryConfig(size=8));colony.expose(cue(1))
    with pytest.raises(ValueError):colony.punish(cue(1),penalty=2)
    colony.freeze()
    with pytest.raises(RuntimeError):colony.punish(cue(1))


def test_reinforce_rewards_correct_and_punishes_wrong():
    model=AssociativeMycelium(('a','b'),MemoryConfig(size=8,colonies_per_label=1))
    a,b=cue(1),cue(2)
    assert model.reinforce(a,'a')['outcome']=='abstain'  # empty memory: teach only
    assert model.reinforce(a,'a')['outcome']=='reward'
    model.learn(b,'b')
    before=model.colonies['a'][0].distance(a)
    out=model.reinforce(a,'b')  # deliberately wrong teacher label for 'a'-like cue
    assert out['prediction']=='a' and out['outcome']=='punish'
    assert model.colonies['a'][0].distance(a)>before
    assert model.reinforce(a,'a')['outcome'] in ('reward','punish')
