"""One-shot drone flight: how many demonstrations does a policy need?

A drone sees a 7x7 egocentric obstacle window plus the direction to its goal
and must fly to the goal on maps it has never seen. Learners are taught from
oracle (shortest-path) demonstrations only; none gets rewards or maps from the
test set. Compared under identical data:

  mycelia  : TraceMemory (kernel-ridge trace recall) over the drone's view
  knn      : 1-nearest-neighbour on the same features
  logreg   : multinomial logistic regression on the same features
  greedy   : no learning, steps toward the goal if free (floor)
  oracle   : BFS shortest path (ceiling)

Also tests sharing: K drones each taught ONE demo on their own map, evaluated
alone vs. with their memories pooled. Pure NumPy + scikit-learn, no GPU.
Research simulation only.
"""
from __future__ import annotations
import argparse, json, time
from collections import deque
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from mycelia.receptor_memory import TraceMemory

TRAIL = True
N, R = 24, 3                          # map size, view radius (7x7 window)
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right


def make_map(rng, density=.22):
    while True:
        g = rng.random((N, N)) < density
        free = np.argwhere(~g)
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        if abs(a - b).sum() < 10:
            continue
        d = bfs(g, tuple(b))
        if d[tuple(a)] >= 0:
            return g, tuple(a), tuple(b), d


def bfs(g, goal):
    d = np.full(g.shape, -1, dtype=np.int32); d[goal] = 0; q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            y, x = r + dr, c + dc
            if 0 <= y < N and 0 <= x < N and not g[y, x] and d[y, x] < 0:
                d[y, x] = d[r, c] + 1; q.append((y, x))
    return d


def features(g, pos, goal, visits=None):
    p = np.pad(g, R, constant_values=True)
    view = p[pos[0]:pos[0] + 2 * R + 1, pos[1]:pos[1] + 2 * R + 1].astype(np.float32).ravel()
    dy, dx = goal[0] - pos[0], goal[1] - pos[1]
    n = max(abs(dy) + abs(dx), 1)
    v = np.zeros((2 * R + 1) ** 2, dtype=np.float32) if visits is None else \
        np.pad(np.minimum(visits, 3) / 3., R)[pos[0]:pos[0] + 2 * R + 1, pos[1]:pos[1] + 2 * R + 1].astype(np.float32).ravel()
    return np.concatenate([view, [dy / N, dx / N, np.sign(dy), np.sign(dx), dy / n, dx / n], v]).astype(np.float32)


def oracle_action(d, pos):
    best, ba = d[pos], -1
    for a, (dr, dc) in enumerate(MOVES):
        y, x = pos[0] + dr, pos[1] + dc
        if 0 <= y < N and 0 <= x < N and d[y, x] >= 0 and d[y, x] < best:
            best, ba = d[y, x], a
    return ba


def demo(rng):
    """One demonstration = one full oracle flight on a fresh map."""
    g, s, goal, d = make_map(rng); X, y, pos = [], [], s; vis = np.zeros((N, N))
    while pos != goal:
        vis[pos] += 1; a = oracle_action(d, pos); X.append(features(g, pos, goal, vis)); y.append(a)
        pos = (pos[0] + MOVES[a][0], pos[1] + MOVES[a][1])
    return np.array(X), np.array(y)


def fly(policy, g, s, goal, d):
    """Return steps used (None if it failed to arrive)."""
    pos, limit, seen = s, 4 * d[s] + 20, {}; vis = np.zeros((N, N))
    for step in range(limit):
        if pos == goal:
            return step
        vis[pos] += 1; f = features(g, pos, goal, vis)[None]
        sc = policy(f)[0]
        for a in np.argsort(-sc):                      # best action that is not a wall
            y, x = pos[0] + MOVES[a][0], pos[1] + MOVES[a][1]
            if 0 <= y < N and 0 <= x < N and not g[y, x]:
                pos = (y, x); break
        seen[pos] = seen.get(pos, 0) + 1
        if seen[pos] > 6:                              # stuck in a loop
            return None
    return None


def greedy_scores(f):
    dy, dx = f[0, 49], f[0, 50]
    sc = np.array([-dy, dy, -dx, dx], dtype=np.float32)
    if TRAIL:
        sc = sc - 1.5 * f[0, [55 + i for i in NEIGHBOR]] * 3 * 0.4
    return sc[None]


NEIGHBOR = (2 * R * 0 + (R - 1) * (2 * R + 1) + R, (R + 1) * (2 * R + 1) + R, R * (2 * R + 1) + R - 1, R * (2 * R + 1) + R + 1)


def prior_actions(F):
    """Greedy goal-seeking prior that respects walls it can see: best free action per row."""
    sc = np.stack([-F[:, 49], F[:, 49], -F[:, 50], F[:, 50]], 1).astype(np.float64)
    if TRAIL:
        sc = sc - 0.6 * F[:, [55 + i for i in NEIGHBOR]]
    for a, idx in enumerate(NEIGHBOR):
        sc[F[:, idx] > .5, a] = -1e9
    return sc.argmax(1)


def corrections(kind, X, y, tau):
    """Prior plus corrections: store only the moments where the prior was wrong."""
    wrong = prior_actions(X) != y
    if not wrong.any():
        return greedy_scores
    Xw, yw = X[wrong], y[wrong]
    mem = TraceMemory(4, gamma=4., lam=.01).fit(Xw, yw) if kind == 'mycelia_corr' else None
    mean = Xw.mean(0); scale = Xw.std(0) + 1e-6
    unit = lambda f: ((f - mean) / scale) / (np.linalg.norm((f - mean) / scale, axis=1, keepdims=True) + 1e-9)
    T = unit(Xw)

    def policy(f):
        sim = unit(f) @ T.T; best = sim.argmax(1)
        base = greedy_scores(f)
        if sim[np.arange(len(f)), best].max() < tau:
            return base
        if mem is not None:
            return mem.scores(f)
        return np.eye(4)[yw[best]]
    return policy


def build(kind, X, y, seed):
    if kind in ('mycelia_corr', 'knn_corr'):
        return corrections(kind, X, y, .85)
    if kind == 'mycelia':
        m = TraceMemory(4, gamma=4., lam=.01).fit(X, y); return m.scores
    if kind == 'knn':
        m = KNeighborsClassifier(1).fit(X, y)
        return lambda f: np.eye(4)[m.predict(f)]
    if kind == 'logreg':
        if len(set(y)) < 2: return lambda f: np.eye(4)[[y[0]] * len(f)]
        m = LogisticRegression(max_iter=300).fit(X, y)
        def scores(f):
            out = np.zeros((len(f), 4)); out[:, m.classes_] = m.predict_proba(f); return out
        return scores
    raise ValueError(kind)


def evaluate(policy, tests):
    ok, ratio = [], []
    for g, s, goal, d in tests:
        r = fly(policy, g, s, goal, d); ok.append(r is not None)
        if r is not None: ratio.append(r / d[s])
    return float(np.mean(ok)), float(np.mean(ratio)) if ratio else None


def ci(vals, rng):
    b = [np.mean(rng.choice(vals, len(vals))) for _ in range(500)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tests', type=int, default=200); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--demos', type=int, nargs='+', default=[1, 2, 5, 10, 25])
    ap.add_argument('--no-trail', action='store_true'); ap.add_argument('--output', type=Path, default=Path('runs/drone-oneshot'))
    a = ap.parse_args(); globals()['TRAIL'] = not a.no_trail; a.output.mkdir(parents=True, exist_ok=True); t0 = time.time()
    res = {'demo_curve': {}, 'sharing': {}, 'config': vars(a) | {'output': str(a.output)}}
    kinds = ('mycelia', 'knn', 'logreg', 'mycelia_corr', 'knn_corr')
    for n in a.demos:
        acc = {k: [] for k in kinds + ('greedy',)}; eff = {k: [] for k in acc}
        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 * seed + 7)
            tests = [make_map(np.random.default_rng(900000 + seed * 1000 + i)) for i in range(a.tests)]
            D = [demo(rng) for _ in range(n)]; X = np.concatenate([d[0] for d in D]); y = np.concatenate([d[1] for d in D])
            for k in kinds:
                s, e = evaluate(build(k, X, y, seed), tests); acc[k].append(s); eff[k].append(e)
            s, e = evaluate(greedy_scores, tests); acc['greedy'].append(s); eff['greedy'].append(e)
        res['demo_curve'][n] = {k: {'success_mean': float(np.mean(v)), 'success_per_seed': v,
                                    'path_ratio_mean': float(np.mean([x for x in eff[k] if x is not None])) if any(x is not None for x in eff[k]) else None}
                                for k, v in acc.items()}
        print(n, 'demos:', {k: round(np.mean(v), 3) for k, v in acc.items()}, f'{time.time()-t0:.0f}s', flush=True)
    for K in (3, 5, 10):                                   # sharing: K drones x ONE demo each
        solo, pooled = [], []
        for seed in range(a.seeds):
            rng = np.random.default_rng(5000 + 1000 * seed + K)
            tests = [make_map(np.random.default_rng(900000 + seed * 1000 + i)) for i in range(a.tests)]
            D = [demo(rng) for _ in range(K)]
            solo.append(np.mean([evaluate(build('mycelia_corr', *D[i], seed), tests)[0] for i in range(K)]))
            pooled.append(evaluate(build('mycelia_corr', np.concatenate([d[0] for d in D]), np.concatenate([d[1] for d in D]), seed), tests)[0])
        res['sharing'][K] = {'solo_one_demo': float(np.mean(solo)), 'pooled_shared': float(np.mean(pooled)),
                             'gain_ci95': ci(np.array(pooled) - np.array(solo), np.random.default_rng(0))}
        print('share', K, res['sharing'][K], flush=True)
    res['seconds'] = time.time() - t0
    (a.output / 'results.json').write_text(json.dumps(res, indent=2))
    print('RESULT_JSON_BEGIN'); print(json.dumps(res)); print('RESULT_JSON_END')


if __name__ == '__main__':
    main()
