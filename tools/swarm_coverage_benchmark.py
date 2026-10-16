"""Swarm coverage: shared mark trail (stigmergy) vs coordinated frontier exploration.

K robots start at one entrance and must visit 95% of the free cells of an unknown 32x32 obstacle map.
Methods (same maps, same start):
  random         : random walk
  marks_private  : each robot avoids cells IT visited (own marks)
  marks_shared   : robots leave decaying marks in the ENVIRONMENT and step to the least-marked
                   neighbour (MYCELIA's trail rule); needs only local reads, no radio
  frontier_shared: each robot walks to the nearest unvisited cell nobody else has claimed, using a
                   global shared map (strong standard baseline; needs full communication)
Scenarios: normal; half_lost (half the robots die at step 100); no_comms (global map unavailable:
frontier robots only know their own visits; environmental marks still work).
Metric: steps to 95% coverage (cap 8000), finish rate. Paired bootstrap marks_shared vs frontier_shared.
CPU only. Research simulation.
"""
from __future__ import annotations
import argparse, json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from collections import deque
from multiprocessing import Pool
from pathlib import Path
import numpy as np

N = 32
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def make_map(seed):
    rng = np.random.default_rng(seed)
    while True:
        g = rng.random((N, N)) < .2
        free = np.argwhere(~g)
        start = tuple(free[rng.integers(len(free))])
        comp = {start}; q = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in MOVES:
                y, x = r + dr, c + dc
                if 0 <= y < N and 0 <= x < N and not g[y, x] and (y, x) not in comp: comp.add((y, x)); q.append((y, x))
        if len(comp) > 0.6 * len(free): return g, start, comp


def neighbors(g, pos):
    for dr, dc in MOVES:
        y, x = pos[0] + dr, pos[1] + dc
        if 0 <= y < N and 0 <= x < N and not g[y, x]: yield (y, x)


def bfs_path(g, src, is_target):
    prev = {src: None}; q = deque([src])
    while q:
        cur = q.popleft()
        if cur != src and is_target(cur):
            path = []
            while cur != src: path.append(cur); cur = prev[cur]
            return path[::-1]
        for nb in neighbors(g, cur):
            if nb not in prev: prev[nb] = cur; q.append(nb)
    return []


def run(method, scenario, K, mapseed, cap=8000):
    g, start, comp = make_map(mapseed); rng = np.random.default_rng(mapseed + 99)
    need = int(.95 * len(comp)); visited = set([start]); pos = [start] * K; alive = [True] * K
    shared = np.zeros((N, N)); priv = [np.zeros((N, N)) for _ in range(K)]; priv_seen = [{start} for _ in range(K)]
    claimed = {}; paths = [[] for _ in range(K)]
    for t in range(1, cap + 1):
        if scenario == 'half_lost' and t == 100:
            for i in range(K // 2): alive[i] = False; claimed.pop(i, None)
        if not any(alive): return None
        shared *= .995
        for i in range(K):
            if not alive[i]: continue
            nbs = list(neighbors(g, pos[i]))
            if not nbs: continue
            if method == 'random': nxt = nbs[int(rng.integers(len(nbs)))]
            elif method in ('marks_shared', 'marks_private'):
                m = shared if method == 'marks_shared' else priv[i]; m[pos[i]] += 1
                lo = min(m[n] for n in nbs); cand = [n for n in nbs if m[n] == lo]; nxt = cand[int(rng.integers(len(cand)))]
            else:                                              # frontier
                comms = scenario != 'no_comms'; seen = visited if comms else priv_seen[i]
                if not paths[i] or paths[i][-1] in seen:
                    taken = {c for j, c in claimed.items() if j != i} if comms else set()
                    paths[i] = bfs_path(g, pos[i], lambda c: c not in seen and c not in taken)
                    claimed[i] = paths[i][-1] if paths[i] else None
                nxt = paths[i].pop(0) if paths[i] else nbs[int(rng.integers(len(nbs)))]
            pos[i] = nxt; visited.add(nxt)
            if method == 'frontier': priv_seen[i].add(nxt)
        if len(visited) >= need: return t
    return None


def job(a):
    method, scenario, K, seed = a
    return (method, scenario, K, seed, run(method, scenario, K, seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--maps', type=int, default=30); ap.add_argument('--ks', type=int, nargs='+', default=[1, 4, 16])
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument('--output', type=Path, default=Path('runs/swarm-coverage'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.time()
    methods = ('random', 'marks_private', 'marks_shared', 'frontier'); scenarios = ('normal', 'half_lost', 'no_comms')
    jobs = [(m, s, K, 5000 + i) for s in scenarios for K in a.ks for m in methods for i in range(a.maps)]
    with Pool(a.workers) as pool: res = pool.map(job, jobs, chunksize=4)
    out = {}; rng = np.random.default_rng(0)
    for s in scenarios:
        for K in a.ks:
            row = {}
            for m in methods:
                v = {seed: t for (mm, ss, kk, seed, t) in res if mm == m and ss == s and kk == K}
                ok = [t for t in v.values() if t is not None]
                row[m] = {'median_steps': float(np.median(ok)) if ok else None, 'finish_rate': len(ok) / len(v)}
            a_ = {seed: t for (mm, ss, kk, seed, t) in res if mm == 'marks_shared' and ss == s and kk == K}
            b_ = {seed: t for (mm, ss, kk, seed, t) in res if mm == 'frontier' and ss == s and kk == K}
            both = [seed for seed in a_ if a_[seed] is not None and b_[seed] is not None]
            if both:
                d = np.array([a_[x] - b_[x] for x in both], dtype=float); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
                row['marks_shared_minus_frontier_steps'] = {'mean': float(d.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], 'n': len(both)}
            out[f'{s}_K{K}'] = row
            print(s, 'K', K, {m: (row[m]['median_steps'], round(row[m]['finish_rate'], 2)) for m in methods}, row.get('marks_shared_minus_frontier_steps'), flush=True)
    (a.output / 'results.json').write_text(json.dumps({'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, 'results': out, 'seconds': time.time() - t0}, indent=2))
    print('RESULT_SUMMARY_BEGIN'); print(json.dumps(out)); print('RESULT_SUMMARY_END')


if __name__ == '__main__':
    main()
