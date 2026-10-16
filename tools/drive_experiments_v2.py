"""Mycelia Autonomous Sandbox v0.2: every condition, 3 seeds x 3 frozen suites,
checked against the pre-registered criterion in AUTONOMY-V2.md."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import defaultdict
from pathlib import Path
from mycelia.autonomy.network import origin
from mycelia.autonomy.network_v2 import run, CONDITIONS
from mycelia.autonomy.sim import Scenario, simulate
from receptor_memory_mri import wilson

SUITES = ['drive_v2_eval_1.json', 'drive_v2_eval_2.json', 'drive_v2_eval_3.json']


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, default=Path('results/autonomy-v2'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    man = json.loads(Path('datasets/drive_v2_manifest.json').read_text())
    for s in SUITES:
        assert hashlib.sha256(Path('datasets', s).read_text().encode()).hexdigest() == man[s]['sha256'], s
    suites = {s: [(Scenario.from_dict({k: v for k, v in d.items() if k not in ('held_out', 'post_shift')}), d['held_out'], d['post_shift'])
                  for d in json.loads(Path('datasets', s).read_text())] for s in SUITES}
    R = {'conditions': {}}
    for cond in CONDITIONS:
        agg = defaultdict(lambda: defaultdict(int)); seeds = []
        for seed, suite in enumerate(SUITES):
            cars, held, stats, memory = run(cond, seed)
            for c in cars:
                pol = c.policy(held[c.name])
                for sc, held_out, post in suites[suite]:
                    res = simulate(sc, pol, brake=c.brake)
                    group = 'held_out' if held_out else ('own_family' if sc.family == c.family else 'other_family')
                    foreign = {origin(r) for r in res['fired']} - {c.name}
                    for g in ('all', group, f'car:{c.name}') + (('post_shift',) if post else ()):
                        d = agg[g]; d['episodes'] += 1; d[res['outcome']] += 1; d['idle_ticks'] += res['idle']
                        d['unsafe_transfer'] += bool(foreign) and res['outcome'] == 'collision'
                        d['liar_rule_fired'] += 'car5' in foreign
            seeds.append({k: v for k, v in stats.items()} | {'coordination_records': len(memory.current('coordination'))})
        m = {g: dict(v, success=v.get('success', 0), collision=v.get('collision', 0), stuck=v.get('stuck', 0),
                     success_rate=v.get('success', 0) / v['episodes'], success_ci=wilson(v.get('success', 0), v['episodes']),
                     collision_rate=v.get('collision', 0) / v['episodes'], collision_ci=wilson(v.get('collision', 0), v['episodes']))
             for g, v in agg.items()}
        sums = lambda key: sum(x.get(key, 0) for x in seeds)
        R['conditions'][cond] = {'metrics': m, 'per_seed': seeds,
                                 'totals': {k: sums(k) for k in ('messages', 'bytes', 'adopted', 'verifications', 'rejected_by_verification',
                                                                  'rejected_by_trust', 'retractions', 'irrelevant_here', 'liar_rules_held_end',
                                                                  'false_rule_acceptance')},
                                 'operational_collisions_by_round': [sum(x['operational_collisions_by_round'][i] for x in seeds) for i in range(4)]}
        al = m['all']
        print(f"{cond:8} success {al['success_rate']:.3f} {[round(x, 3) for x in al['success_ci']]} collisions {al['collision']} "
              f"{[round(x, 4) for x in al['collision_ci']]} unsafe-transfer {al['unsafe_transfer']} liar-fired {al['liar_rule_fired']} "
              f"idle {al['idle_ticks']} ops/round {R['conditions'][cond]['operational_collisions_by_round']} {R['conditions'][cond]['totals']}", flush=True)
    d, b = R['conditions']['D_full']['metrics']['all'], R['conditions']['B_scope']['metrics']['all']
    better_col = d['collision_ci'][1] < b['collision_ci'][0]; worse_col = d['collision_ci'][0] > b['collision_ci'][1]
    better_succ = d['success_ci'][0] > b['success_ci'][1]; worse_succ = d['success_ci'][1] < b['success_ci'][0]
    R['primary_criterion'] = {'D_full_better_collisions': better_col, 'D_full_better_success': better_succ,
                              'D_full_worse_collisions': worse_col, 'D_full_worse_success': worse_succ,
                              'adaptive_helps': (better_col and not worse_succ) or (better_succ and not worse_col)}
    R['seconds'] = time.monotonic() - t0; print('primary criterion', R['primary_criterion'])
    (a.output / 'drive-v2-results.json').write_text(json.dumps(R, indent=1, default=str))


if __name__ == '__main__':
    main()
