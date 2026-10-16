"""Drone and robotics battery: where does the mark-colony earn its keep?

E1  Learner showdown (same flight budget, same maps): reward-driven colony
    (cross-entropy search over a mark-aware policy) vs tabular Q-learning, with
    and without access to marks.
E2  Discovering marking: the drone starts with NO deposit (marks off) and must
    learn both to drop marks and what they mean.
E3  Fleet coverage (robotics): K robots explore an unknown map. Shared marks
    (one common trail) vs private marks vs random walk; then half the fleet
    is lost mid-run (damage tolerance).

All grids/NumPy, no GPU. Research simulation only; not flight software.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from drone_oneshot_flight import N, MOVES, make_map
from drone_learn_trail import fly as fly_marks, test as test_marks


# ---------- E1: tabular Q-learning baseline ----------
def q_state(g, pos, goal, vis):
    sy, sx = int(np.sign(goal[0] - pos[0])), int(np.sign(goal[1] - pos[1]))
    free, v = [], []
    for dr, dc in MOVES:
        y, x = pos[0] + dr, pos[1] + dc
        ok = 0 <= y < N and 0 <= x < N and not g[y, x]
        free.append(ok); v.append(min(int(vis[y, x]), 2) if ok else 0)
    return (sy, sx, tuple(free), tuple(v)), free


def q_episode(Q, g, s, goal, d, eps, alpha=.3, gamma=.95, learn=True, marks=True, rng=None):
    pos, limit, vis = s, 4 * d[s] + 20, np.zeros((N, N))
    for step in range(limit):
        if pos == goal: return step
        vis[pos] += 1
        st, free = q_state(g, pos, goal, vis if marks else np.zeros((N, N)))
        acts = [a for a in range(4) if free[a]]
        if not acts: return None
        q = Q.setdefault(st, np.zeros(4))
        a = acts[int(rng.integers(len(acts)))] if learn and rng.random() < eps else max(acts, key=lambda k: q[k])
        nxt = (pos[0] + MOVES[a][0], pos[1] + MOVES[a][1])
        r = 1. if nxt == goal else -.01
        if learn:
            st2, free2 = q_state(g, nxt, goal, vis if marks else np.zeros((N, N)))
            q2 = Q.setdefault(st2, np.zeros(4)); nxtbest = 0. if nxt == goal else max(q2[k] for k in range(4) if free2[k]) if any(free2) else 0.
            q[a] += alpha * (r + gamma * nxtbest - q[a])
        pos = nxt
    return None


def run_q(seed, marks, flights, tests):
    rng = np.random.default_rng(seed); Q = {}; curve = []
    for f in range(flights + 1):
        if f % 960 == 0:
            ok = [q_episode(Q, *t, 0, learn=False, marks=marks, rng=rng) is not None for t in tests]
            curve.append({'flights': f, 'test_success': float(np.mean(ok))})
        if f == flights: break
        q_episode(Q, *make_map(rng), eps=max(.05, .5 * (1 - f / flights)), marks=marks, rng=rng)
    return curve


# ---------- E2: learn to deposit AND to avoid marks ----------
def fly_deposit(w, g, s, goal, d):
    pos, limit, marks = s, 4 * d[s] + 20, np.zeros((N, N)); dep = max(float(w[3]), 0.); seen = {}
    for step in range(limit):
        if pos == goal: return step
        marks *= float(np.clip(w[2], 0, .99)); marks[pos] += dep
        best, bs = None, -1e18
        for dr, dc in MOVES:
            y, x = pos[0] + dr, pos[1] + dc
            if 0 <= y < N and 0 <= x < N and not g[y, x]:
                prog = (abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])) - (abs(goal[0] - y) + abs(goal[1] - x))
                sc = w[0] * prog + w[1] * marks[y, x]
                if sc > bs: bs, best = sc, (y, x)
        if best is None: return None
        pos = best; seen[pos] = seen.get(pos, 0) + 1
        if seen[pos] > 8: return None
    return None


def run_discover(seed, gens, pop, elite, flights, tests):
    rng = np.random.default_rng(seed); mean = np.array([1., 0., .5, 0.]); sigma = np.array([.5, .8, .2, .6]); curve = []
    ev = lambda w: float(np.mean([fly_deposit(w, *t) is not None for t in tests]))
    for gen in range(gens + 1):
        if gen % 5 == 0: curve.append({'flights': gen * pop * flights, 'test_success': ev(mean), 'w': mean.round(3).tolist()})
        if gen == gens: break
        maps = [make_map(rng) for _ in range(flights)]; cand = mean + sigma * rng.standard_normal((pop, 4))
        sc = []
        for c in cand:
            r = 0.
            for m in maps:
                k = fly_deposit(c, *m); r += (1 - .004 * k) if k is not None else -.2
            sc.append(r / len(maps))
        top = cand[np.argsort(-np.array(sc))[:elite]]
        mean = .5 * mean + .5 * top.mean(0); sigma = .5 * sigma + .5 * (top.std(0) + .05)
    return curve


# ---------- E3: fleet coverage ----------
def coverage(K, mode, g, rng, target=.9, kill_at=None, cap=4000, w_mark=-1.5):
    free = np.argwhere(~g); need = int(target * len(free)); seen = np.zeros((N, N), bool)
    pos = [tuple(free[rng.integers(len(free))]) for _ in range(K)]; alive = [True] * K
    shared = np.zeros((N, N)); priv = [np.zeros((N, N)) for _ in range(K)]
    for t in range(cap):
        if kill_at is not None and t == kill_at:
            for i in range(K // 2): alive[i] = False
        if not any(alive): return None
        for i in range(K):
            if not alive[i]: continue
            m = shared if mode == 'shared' else priv[i]
            cells = []
            for dr, dc in MOVES:
                y, x = pos[i][0] + dr, pos[i][1] + dc
                if 0 <= y < N and 0 <= x < N and not g[y, x]: cells.append((y, x))
            if not cells: continue
            if mode == 'random': pos[i] = cells[int(rng.integers(len(cells)))]
            else:
                m[pos[i]] += 1; pos[i] = min(cells, key=lambda c: (m[c], rng.random())) if w_mark < 0 else cells[0]
            seen[pos[i]] = True
        if seen.sum() >= need: return t
    return None


def run_fleet(seeds, maps_per_seed):
    out = {}
    for K in (1, 2, 4, 8):
        for mode in ('random', 'private', 'shared'):
            ts = []
            for sd in range(seeds):
                for j in range(maps_per_seed):
                    rng = np.random.default_rng(sd * 100 + j); g = make_map(rng)[0]
                    ts.append(coverage(K, mode, g, np.random.default_rng(sd * 100 + j + 7)))
            ok = [t for t in ts if t is not None]
            out[f'K{K}_{mode}'] = {'median_steps': float(np.median(ok)) if ok else None, 'finished': len(ok) / len(ts)}
        killed = []
        for sd in range(seeds):
            for j in range(maps_per_seed):
                rng = np.random.default_rng(sd * 100 + j); g = make_map(rng)[0]
                killed.append(coverage(K, 'shared', g, np.random.default_rng(sd * 100 + j + 7), kill_at=40) if K > 1 else None)
        ok = [t for t in killed if t is not None]
        if K > 1: out[f'K{K}_shared_half_lost_at_40'] = {'median_steps': float(np.median(ok)) if ok else None, 'finished': len(ok) / len(killed)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--tests', type=int, default=100)
    ap.add_argument('--output', type=Path, default=Path('runs/drone-robotics-battery'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.time(); res = {}
    e1 = {'q_marks': [], 'q_no_marks': [], 'colony_marks': []}
    for seed in range(a.seeds):
        tests = [make_map(np.random.default_rng(900000 + seed * 1000 + i)) for i in range(a.tests)]
        e1['q_marks'].append(run_q(seed, True, 5760, tests)); e1['q_no_marks'].append(run_q(seed, False, 5760, tests))
        from drone_learn_trail import learn
        e1['colony_marks'].append(learn(seed, True, 30, 24, 6, 8, tests))
        print('E1 seed', seed, {k: round(v[-1][-1]['test_success'] if isinstance(v[-1], list) else v[-1]['test_success'], 3) for k, v in ((k, [e1[k][-1]]) for k in e1)}, f'{time.time()-t0:.0f}s', flush=True)
    res['E1_showdown'] = e1
    res['E1_final_mean'] = {k: float(np.mean([c[-1]['test_success'] for c in v])) for k, v in e1.items()}
    print('E1 final', res['E1_final_mean'], flush=True)
    e2 = [run_discover(s, 40, 24, 6, 8, [make_map(np.random.default_rng(900000 + s * 1000 + i)) for i in range(a.tests)]) for s in range(a.seeds)]
    res['E2_discover_marking'] = e2
    res['E2_final'] = {'start': float(np.mean([c[0]['test_success'] for c in e2])), 'final': float(np.mean([c[-1]['test_success'] for c in e2])),
                       'deposit_final': [c[-1]['w'][3] for c in e2], 'w_mark_final': [c[-1]['w'][1] for c in e2]}
    print('E2', res['E2_final'], f'{time.time()-t0:.0f}s', flush=True)
    res['E3_fleet'] = run_fleet(a.seeds, 4)
    for k, v in res['E3_fleet'].items(): print('E3', k, v, flush=True)
    res['seconds'] = time.time() - t0
    (a.output / 'results.json').write_text(json.dumps(res, indent=2)); print('done', f'{res["seconds"]:.0f}s')


if __name__ == '__main__':
    main()
