"""Can the drone learn the trail rule from its own flights?

The drone is given a mark memory (it deposits a decaying mark on every cell it
flies over) but NOT told what marks mean. Its policy scores each free
neighbour as   w_goal * progress_toward_goal + w_mark * mark(neighbour).
Both weights start so that marks are ignored (w_mark = 0). Learning is
reward-driven: a colony of candidate "tips" (weight vectors) flies fresh maps,
the best-rewarded tips are reinforced and the rest are pruned (a
cross-entropy-method search). Reward = arrive (+1) minus a small step cost.

Controls on identical maps and budget:
  marks      : may learn to use marks (the experiment)
  no-marks   : marks hidden from the policy (isolates what marks add)
  fixed rule : hand-coded trail penalty from drone_oneshot_flight (ceiling)

Success if learned w_mark < 0 (avoid your own trail) and test success rises.
Pure NumPy, no GPU. Research simulation only.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from drone_oneshot_flight import N, MOVES, make_map


def fly(w, g, s, goal, d, use_marks=True):
    """Fly with weights w = (w_goal, w_mark, decay). Returns steps or None."""
    pos, limit = s, 4 * d[s] + 20
    marks = np.zeros((N, N)); decay = float(np.clip(w[2], 0., .99))
    seen = {}
    for step in range(limit):
        if pos == goal:
            return step
        marks *= decay; marks[pos] += 1.
        best, bs = None, -1e18
        for dr, dc in MOVES:
            y, x = pos[0] + dr, pos[1] + dc
            if 0 <= y < N and 0 <= x < N and not g[y, x]:
                progress = (abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])) - (abs(goal[0] - y) + abs(goal[1] - x))
                sc = w[0] * progress + (w[1] * marks[y, x] if use_marks else 0.)
                if sc > bs: bs, best = sc, (y, x)
        if best is None:
            return None
        pos = best; seen[pos] = seen.get(pos, 0) + 1
        if seen[pos] > 8:
            return None
    return None


def reward(w, maps, use_marks):
    r = 0.
    for g, s, goal, d in maps:
        steps = fly(w, g, s, goal, d, use_marks); r += (1. - .004 * steps) if steps is not None else -.2
    return r / len(maps)


def test(w, tests, use_marks):
    ok = [fly(w, *t, use_marks) is not None for t in tests]; return float(np.mean(ok))


def learn(seed, use_marks, gens, pop, elite, flights, tests):
    rng = np.random.default_rng(seed)
    mean = np.array([1., 0., .5]); sigma = np.array([.5, .8, .2]); curve = []
    for gen in range(gens + 1):
        if gen % 5 == 0:
            curve.append({'flights': gen * pop * flights, 'test_success': test(mean, tests, use_marks), 'w': mean.round(3).tolist()})
        if gen == gens: break
        maps = [make_map(rng) for _ in range(flights)]                  # fresh experience each generation
        cand = mean + sigma * rng.standard_normal((pop, 3))
        scores = np.array([reward(c, maps, use_marks) for c in cand])
        top = cand[np.argsort(-scores)[:elite]]                         # reinforce winners, prune the rest
        mean = .5 * mean + .5 * top.mean(0); sigma = .5 * sigma + .5 * (top.std(0) + .05)
    return curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--gens', type=int, default=30)
    ap.add_argument('--pop', type=int, default=24); ap.add_argument('--elite', type=int, default=6)
    ap.add_argument('--flights', type=int, default=8); ap.add_argument('--tests', type=int, default=100)
    ap.add_argument('--output', type=Path, default=Path('runs/drone-learn-trail'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.time()
    out = {'config': {k: v for k, v in vars(a).items() if k != 'output'}, 'seeds': []}
    for seed in range(a.seeds):
        tests = [make_map(np.random.default_rng(900000 + seed * 1000 + i)) for i in range(a.tests)]
        row = {'seed': seed,
               'marks': learn(seed, True, a.gens, a.pop, a.elite, a.flights, tests),
               'no_marks': learn(seed, False, a.gens, a.pop, a.elite, a.flights, tests),
               'fixed_rule_reference': test(np.array([1., -.5, .9]), tests, True)}
        out['seeds'].append(row)
        print(f"seed {seed}: marks {row['marks'][-1]['test_success']:.3f} (w={row['marks'][-1]['w']}) "
              f"no-marks {row['no_marks'][-1]['test_success']:.3f} start {row['marks'][0]['test_success']:.3f} "
              f"fixed {row['fixed_rule_reference']:.3f} {time.time()-t0:.0f}s", flush=True)
    fin = lambda k: [s[k][-1]['test_success'] for s in out['seeds']]
    out['summary'] = {'start': float(np.mean([s['marks'][0]['test_success'] for s in out['seeds']])),
                      'marks_final': float(np.mean(fin('marks'))), 'no_marks_final': float(np.mean(fin('no_marks'))),
                      'w_mark_final': [s['marks'][-1]['w'][1] for s in out['seeds']],
                      'fixed_rule': float(np.mean([s['fixed_rule_reference'] for s in out['seeds']]))}
    out['seconds'] = time.time() - t0
    (a.output / 'results.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out['summary']))


if __name__ == '__main__':
    main()
