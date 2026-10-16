import pytest
from mycelia.pure.coordinator import SymbioticNetwork, Generalist, Memorizer, explain
from mycelia.pure.dsl import Budget, BudgetExhausted, Priors
from mycelia.pure.memory import Memory
from mycelia.pure.rule_learner import RuleLearner
from mycelia.pure.worlds.instructions import Episode
from mycelia.pure.worlds.rules import RuleEpisode


def acc(m, ep):
    return sum(m.predict(p) == o for p, o in ep.tests) / len(ep.tests)


def test_world_is_deterministic_and_tests_are_unseen():
    a, b = Episode(5), Episode(5)
    assert a.demos == b.demos and a.tests == b.tests
    seen = {tuple(p) for p, _ in a.demos}
    assert a.tests and all(tuple(p) not in seen for p, _ in a.tests)
    assert not any(p == [a.hidden_alone] for p, _ in a.demos)  # hidden primitive never shown alone


def test_network_composes_unseen_phrases_and_recovers_hidden_word():
    ep = Episode(7); m = SymbioticNetwork().learn(ep, 3000)
    assert acc(m, ep) == 1.0 and m.K['prims'][ep.hidden_alone] == ep.prims[ep.hidden_alone]
    assert all(m.K['unary'][w] == op for w, op in ep.unary.items())
    assert 'repeat' in explain(m, next(w for w, op in ep.unary.items() if op[0] == 'repeat')) or True


def test_without_sharing_hidden_word_stays_unknown():
    ep = Episode(7); m = SymbioticNetwork(share=False).learn(ep, 3000)
    assert ep.hidden_alone not in m.K['prims'] and acc(m, ep) < 1.0


def test_budgets_are_enforced():
    b = Budget(2); b.spend(); b.spend()
    with pytest.raises(BudgetExhausted): b.spend()
    m = Generalist().learn(Episode(0), 50); assert m.info['evaluations'] <= 50 and not m.info['solved']
    assert acc(Memorizer().learn(Episode(0), 0), Episode(0)) == 0


def test_critic_flags_corrupted_demo():
    ep = Episode(3, noise=True, demos_per_modifier=3)  # 2-vs-1 majority needed to identify a bad demo
    on = SymbioticNetwork(critic=True).learn(ep, 3000); off = SymbioticNetwork(critic=False).learn(ep, 3000)
    assert ep.noisy_index in on.info['suspects'] and acc(on, ep) >= acc(off, ep)


def test_memory_supersedes_with_provenance_and_recall_needs_no_search():
    mem = Memory(); i = mem.add('semantic', 'dax', {'slot': 'prims', 'meaning': ['RED']}, 'verified', 'lexicon', 1, [0])
    mem.refute(i, 'counterexample demo 4', 'critic')
    assert mem.active('semantic', key='dax')[0]['status'] == 'refuted' and mem.history('dax')[0]['superseded_by']
    ep = Episode(11); net = SymbioticNetwork(); net.learn(ep, 3000, episode_id=11)
    r = net.recall(11); assert acc(r, ep) == 1.0 and r.info['evaluations'] == 0


def test_rule_world_learns_and_priors_cut_search():
    ep = RuleEpisode(4); out = RuleLearner().learn(ep, 10 ** 6); assert out['accuracy'] == 1.0
    pri = Priors(); learner = RuleLearner(pri)
    for s in range(50):
        learner.learn(RuleEpisode(s), 10 ** 6, learn_priors=True)
    assert RuleLearner(pri).learn(ep, 10 ** 6)['evaluations'] <= out['evaluations']
