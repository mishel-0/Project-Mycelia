"""Mycelia Autonomous Sandbox v0.1 experiment: does adaptive symbiosis help
cars handle scenarios more reliably than isolated learning, a shared rule
repository, or fixed sharing? Eval suites were frozen (SHA-256) before the
learners were written. Every car drives every scenario of each suite."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import defaultdict
from pathlib import Path
from mycelia.autonomy.network import build, origin
from mycelia.autonomy.sim import Scenario, simulate
from mycelia.autonomy.scenarios import TRAIN_FAMILIES
from receptor_memory_mri import wilson

SUITES = ['drive_eval_1.json', 'drive_eval_2.json', 'drive_eval_3.json']


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--budget', type=int, default=300)
    ap.add_argument('--conditions', default='A,B,C,D,A_scope,B_scope')
    ap.add_argument('--output', type=Path, default=Path('results/autonomy')); a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path('datasets/drive_manifest.json').read_text()); t0 = time.monotonic(); R = {'budget_episodes_per_car': a.budget, 'runs': {}}
    for s in SUITES:
        assert hashlib.sha256(Path('datasets', s).read_text().encode()).hexdigest() == man[s]['sha256'], s
    suites = {s: [Scenario.from_dict({k: v for k, v in d.items() if k != 'held_out'}) for d in json.loads(Path('datasets', s).read_text())] for s in SUITES}
    for cond in a.conditions.split(','):
        agg = defaultdict(lambda: defaultdict(int)); runs = []
        for seed, suite in enumerate(SUITES):
            cars, extras, stats, memory = build(cond, a.budget, seed)
            rules = {c.name: [(r['cond'], r['action'], r['scope']) for r in c.rules] for c in cars}
            for c in cars:
                pol = c.policy(extras[c.name])
                for sc in suites[suite]:
                    res = simulate(sc, pol)
                    group = 'own_family' if sc.family == c.family else ('held_out' if sc.family not in TRAIN_FAMILIES.values() else 'other_cars_family')
                    for g in (group, 'all', f'family:{sc.family}'):
                        d = agg[g]; d['episodes'] += 1; d[res['outcome']] += 1; d['unnecessary_stops'] += res['unnecessary_stops']
                        d['idle_ticks'] += res['idle']
                        foreign = {origin(r) for r in res['fired']} - {c.name}
                        d['foreign_rule_used'] += bool(foreign)
                        d['incorrect_transfer'] += bool(foreign) and res['outcome'] == 'collision'
                        d['helpful_transfer'] += bool(foreign) and res['outcome'] == 'success'
            runs.append({'suite': suite, 'rules': rules, 'sharing': stats, 'coordination_records': len(memory.current('coordination'))})
        R['runs'][cond] = {'per_seed': runs, 'metrics': {g: dict(v, success_rate=v['success'] / v['episodes'],
                                                                 success_wilson_95=wilson(v['success'], v['episodes']),
                                                                 collision_rate=v['collision'] / v['episodes'], collision=v['collision'], stuck=v['stuck'])
                                                         for g, v in agg.items()}}
        m = R['runs'][cond]['metrics']
        print(cond, {g: (round(m[g]['success_rate'], 3), m[g].get('collision', 0), m[g].get('stuck', 0)) for g in ('own_family', 'other_cars_family', 'held_out', 'all')},
              'incorrect transfers', m['all'].get('incorrect_transfer', 0), 'helpful', m['all'].get('helpful_transfer', 0), flush=True)
    R['seconds'] = time.monotonic() - t0
    (a.output / 'drive-results.json').write_text(json.dumps(R, indent=1, default=str))


if __name__ == '__main__':
    main()
