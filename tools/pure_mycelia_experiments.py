"""Pure Mycelia experiments (no neural networks, no pretrained models, no LLMs).

Splits by seed: train episodes 0-199 (only used to learn search priors),
test episodes 10000+ (fresh random vocabularies and hidden rules, scored once).
Every condition in a comparison gets the same evaluation budget.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from mycelia.pure.coordinator import SymbioticNetwork, Generalist, Vote, Memorizer
from mycelia.pure.dsl import Priors
from mycelia.pure.memory import Memory
from mycelia.pure.rule_learner import RuleLearner
from mycelia.pure.worlds.instructions import Episode
from mycelia.pure.worlds.rules import RuleEpisode
from receptor_memory_mri import wilson

TRAIN, TEST = range(0, 200), range(10000, 10200)


def score(learner, episodes, budget, **kw):
    correct = total = 0; evals = []; solved = 0
    for ep in episodes:
        m = learner.learn(ep, budget, **kw); hits = sum(m.predict(p) == o for p, o in ep.tests)
        correct += hits; total += len(ep.tests); evals.append(m.info['evaluations']); solved += hits == len(ep.tests)
    return {'accuracy': correct / total, 'wilson_95': wilson(correct, total), 'phrases': total,
            'episodes_fully_solved': solved, 'episodes': len(episodes), 'mean_evaluations': float(np.mean(evals))}


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, default=Path('results/pure'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); start = time.monotonic(); R = {}
    test = [Episode(s) for s in TEST]

    # A. Symbiosis: equal budgets.
    R['symbiosis'] = []
    for budget in (30, 100, 300, 1000, 3000, 10000, 30000, 100000):
        row = {'budget': budget}
        for name, make in (('network', SymbioticNetwork), ('network_no_sharing', lambda: SymbioticNetwork(share=False)),
                           ('single_generalist', Generalist), ('vote_of_3', Vote), ('memorizer', Memorizer)):
            row[name] = score(make(), test, budget)
        R['symbiosis'].append(row); print('A', budget, {k: round(v['accuracy'], 3) for k, v in row.items() if k != 'budget'}, flush=True)

    # B. Procedural memory: priors learned on train episodes only.
    priors = Priors(); trainer = SymbioticNetwork(priors=priors)
    for s in TRAIN:
        trainer.learn(Episode(s), 10 ** 6, learn_priors=True)
    R['procedural_memory'] = {'learned_priors': priors.counts, 'rows': []}
    for budget in (20, 40, 60, 100, 10 ** 6):
        row = {'budget': budget, 'without_priors': score(SymbioticNetwork(), test, budget),
               'with_priors': score(SymbioticNetwork(priors=Priors(priors.counts)), test, budget)}
        R['procedural_memory']['rows'].append(row); print('B', budget, row['without_priors']['accuracy'], row['with_priors']['accuracy'],
                                                          row['without_priors']['mean_evaluations'], row['with_priors']['mean_evaluations'], flush=True)

    # C. Critic with one corrupted demonstration per episode.
    noisy = [Episode(s, noise=True, demos_per_modifier=3) for s in TEST]; R['critic'] = {}  # 3 demos so a majority exists
    for name, critic in (('critic_on', True), ('critic_off', False)):
        res = score(SymbioticNetwork(critic=critic), noisy, 3000); found = 0
        for ep in noisy:
            m = SymbioticNetwork(critic=critic).learn(ep, 3000); found += ep.noisy_index in m.info['suspects']
        res['corrupted_demo_flagged'] = found; R['critic'][name] = res; print('C', name, res['accuracy'], found, flush=True)

    # D. Retention: rebuild from verified memory with zero search.
    mem = Memory(); net = SymbioticNetwork(memory=mem); correct = total = 0
    for i, ep in enumerate(test):
        net.learn(ep, 3000, episode_id=i)
    for i, ep in enumerate(test):
        m = net.recall(i); correct += sum(m.predict(p) == o for p, o in ep.tests); total += len(ep.tests)
    R['retention'] = {'accuracy_from_memory_only': correct / total, 'evaluations': 0,
                      'semantic_facts': len(mem.active('semantic')), 'failure_records': len(mem.active('failure'))}
    print('D', R['retention'], flush=True)

    # E. Sample efficiency: fewer demonstrations per modifier.
    R['sample_efficiency'] = []
    for k in (1, 2, 3):
        eps = [Episode(s, demos_per_modifier=k) for s in TEST]
        R['sample_efficiency'].append({'demos_per_modifier': k, 'network_3000': score(SymbioticNetwork(), eps, 3000),
                                       'generalist_3000': score(Generalist(), eps, 3000),
                                       'generalist_300000': score(Generalist(), eps, 300000)})
        print('E', k, {n: round(v['accuracy'], 3) for n, v in R['sample_efficiency'][-1].items() if n != 'demos_per_modifier'}, flush=True)

    # F. Rule world: learned priors and critic.
    rp = Priors(); learner = RuleLearner(rp)
    for s in TRAIN:
        learner.learn(RuleEpisode(s), 10 ** 6, learn_priors=True)
    rtest = [RuleEpisode(s) for s in TEST]; R['rule_world'] = {'learned_priors': rp.counts, 'rows': []}
    for budget in (10, 50, 200, 1000, 10 ** 6):
        row = {'budget': budget}
        for name, L in (('without_priors', RuleLearner()), ('with_priors', RuleLearner(Priors(rp.counts)))):
            out = [L.learn(ep, budget) for ep in rtest]
            row[name] = {'mean_test_accuracy': float(np.mean([o['accuracy'] or 0 for o in out])),
                         'rule_exact': float(np.mean([o['rule'] == ep.rule or (o['accuracy'] == 1.) for o, ep in zip(out, rtest)])),
                         'mean_evaluations': float(np.mean([o['evaluations'] for o in out]))}
        R['rule_world']['rows'].append(row); print('F', budget, row, flush=True)
    rnoisy = [RuleEpisode(s, noise=True) for s in TEST]; R['rule_world']['noise'] = {}
    for name, crit in (('critic_on', True), ('critic_off', False)):
        out = [RuleLearner(Priors(rp.counts), critic=crit).learn(ep, 10 ** 6) for ep in rnoisy]
        R['rule_world']['noise'][name] = {'mean_test_accuracy': float(np.mean([o['accuracy'] or 0 for o in out])),
                                          'corrupted_flagged': int(sum(o['outlier'] == ep.noisy_index for o, ep in zip(out, rnoisy)))}
    print('F noise', R['rule_world']['noise'], flush=True)

    # G. Communication conventions.
    stats = {'full_bytes': 0, 'symbol_bytes': 0, 'symbols': 0, 'messages': 0}
    for ep in test[:50]:
        m = SymbioticNetwork().learn(ep, 3000)
        for k in ('full_bytes', 'symbol_bytes'):
            stats[k] += m.info['bus'][k]
        stats['symbols'] += m.info['symbols']; stats['messages'] += m.info['messages']
    R['conventions'] = stats; R['seconds'] = time.monotonic() - start
    (a.output / 'pure-results.json').write_text(json.dumps(R, indent=2, default=str)); print('G', stats, flush=True)


if __name__ == '__main__':
    main()
