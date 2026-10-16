"""Mycelia Autonomous Sandbox v0.3: bio-inspired mechanisms vs the best v0.2
system (M3) and isolated cars, with one-mechanism-removed ablations, per-obstacle
results and a practice-budget curve. Checked against AUTONOMY-V3.md."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import defaultdict
from pathlib import Path
from mycelia.autonomy import bio, network_v2
from mycelia.autonomy.network import origin
from mycelia.autonomy.sim import Scenario, simulate
from receptor_memory_mri import wilson

SUITES = ['drive_v3_eval_1.json', 'drive_v3_eval_2.json', 'drive_v3_eval_3.json']
MECHS = sorted(bio.ALL)


def conditions():
    out = {'A': lambda s, b: network_v2.run('A', s, budget=b)[:3], 'M3': lambda s, b: network_v2.run('M3', s, budget=b)[:3],
           'BIO_full': lambda s, b: bio.run(bio.ALL, s, budget=b)[:3]}
    for m in MECHS:
        out[f'BIO_without_{m}'] = (lambda m: lambda s, b: bio.run(bio.ALL - {m}, s, budget=b)[:3])(m)
    return out


def evaluate(make, suites, budget):
    agg = defaultdict(lambda: defaultdict(int)); per_seed = []
    for seed, suite in enumerate(SUITES):
        cars, held, stats = make(seed, budget)
        for c in cars:
            pol = c.policy(held[c.name])
            for sc, held_out, post, obstacles in suites[suite]:
                res = simulate(sc, pol, brake=c.brake)
                group = 'held_out' if held_out else ('own_family' if sc.family == c.family else 'other_family')
                foreign = {origin(r) for r in res['fired']} - {c.name}
                for g in ['all', group, f'car:{c.name}'] + [f'obstacle:{o}' for o in obstacles]:
                    d = agg[g]; d['episodes'] += 1; d[res['outcome']] += 1; d['idle_ticks'] += res['idle']
                    d['unsafe_transfer'] += bool(foreign) and res['outcome'] == 'collision'; d['liar_rule_fired'] += 'car5' in foreign
        per_seed.append(stats)
    return {g: dict(v, success=v.get('success', 0), collision=v.get('collision', 0), stuck=v.get('stuck', 0),
                    success_rate=v.get('success', 0) / v['episodes'], success_ci=wilson(v.get('success', 0), v['episodes']),
                    collision_rate=v.get('collision', 0) / v['episodes'], collision_ci=wilson(v.get('collision', 0), v['episodes']))
            for g, v in agg.items()}, per_seed


def separates(a, b, key, lower_is_better):
    """True if a is better than b on key with non-overlapping intervals."""
    return a[key][1] < b[key][0] if lower_is_better else a[key][0] > b[key][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', type=Path, default=Path('results/autonomy-v3'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    man = json.loads(Path('datasets/drive_v3_manifest.json').read_text())
    for s in SUITES:
        assert hashlib.sha256(Path('datasets', s).read_text().encode()).hexdigest() == man[s]['sha256'], s
    suites = {s: [(Scenario.from_dict({k: v for k, v in d.items() if k not in ('held_out', 'post_shift', 'obstacles')}), d['held_out'],
                   d['post_shift'], d['obstacles']) for d in json.loads(Path('datasets', s).read_text())] for s in SUITES}
    R = {'conditions': {}, 'budget_curve': {}}
    for name, make in conditions().items():
        m, seeds = evaluate(make, suites, 400)
        R['conditions'][name] = {'metrics': m, 'per_seed': seeds}
        al = m['all']; obs = {o: (round(m[f'obstacle:{o}']['success_rate'], 3), m[f'obstacle:{o}']['collision'])
                              for o in ('block', 'pedestrian', 'vehicle', 'none') if f'obstacle:{o}' in m}
        print(f"{name:22} success {al['success_rate']:.3f} {[round(x, 3) for x in al['success_ci']]} collisions {al['collision']} "
              f"{[round(x, 3) for x in al['collision_ci']]} liar-fired {al['liar_rule_fired']} per-obstacle {obs}", flush=True)
    for name in ('A', 'M3', 'BIO_full'):
        for b in (100, 200):
            m, _ = evaluate(conditions()[name], suites, b)
            R['budget_curve'].setdefault(name, {})[b] = {'success_rate': m['all']['success_rate'], 'collision': m['all']['collision']}
        R['budget_curve'][name][400] = {k: R['conditions'][name]['metrics']['all'][k] for k in ('success_rate', 'collision')}
        print('curve', name, {b: (round(v['success_rate'], 3), v['collision']) for b, v in R['budget_curve'][name].items()}, flush=True)
    full, m3 = R['conditions']['BIO_full']['metrics']['all'], R['conditions']['M3']['metrics']['all']
    better = separates(full, m3, 'collision_ci', True) or separates(full, m3, 'success_ci', False)
    worse = separates(m3, full, 'collision_ci', True) or separates(m3, full, 'success_ci', False)
    R['primary'] = {'BIO_full_beats_M3': better and not worse, 'BIO_full_better_somewhere': better, 'BIO_full_worse_somewhere': worse}
    R['contributions'] = {}
    for mname in MECHS:
        ab = R['conditions'][f'BIO_without_{mname}']['metrics']['all']
        R['contributions'][mname] = {'removing_it_hurts': separates(full, ab, 'collision_ci', True) or separates(full, ab, 'success_ci', False),
                                     'removing_it_helps': separates(ab, full, 'collision_ci', True) or separates(ab, full, 'success_ci', False)}
    print('primary', R['primary']); print('contributions', R['contributions'])
    R['seconds'] = time.monotonic() - t0
    (a.output / 'drive-v3-results.json').write_text(json.dumps(R, indent=1, default=str))


if __name__ == '__main__':
    main()
