"""Perception + navigation: MYCELIA receptor vision vs a small CNN vs raw-pixel SVM.

Every grid cell has a camera patch (16x16 grayscale) of one of three textures:
floor, wall, hazard (random phase/frequency/contrast + noise). A drone must
fly to its goal on unseen maps using only what it PERCEIVES from patches:
walking into a hazard is a failure, bumping a mis-seen wall costs a step (and
teaches the drone that cell is solid). Navigation uses a mark trail with the
avoid-your-own-trail rule the drone learned earlier (fixed weights here, so
only perception differs between methods).

Compared at several label budgets and camera-noise levels, with identical
labelled data: MYCELIA (unsupervised receptor field + teach-only trace memory),
a small CNN (PyTorch, CPU), a raw-pixel SVM, and ground truth (ceiling).
Reports perception accuracy, task success, and per-cell compute.
Research simulation only.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from drone_oneshot_flight import N, MOVES, make_map, bfs
from mycelia.receptor_memory import ReceptorConfig, ReceptorField, TraceMemory

S = 16
FLOOR, WALL, HAZARD = 0, 1, 2
yy, xx = np.mgrid[0:S, 0:S] / S


def render(cls, rng, noise):
    """Procedural camera patch for a cell class (random pose each time)."""
    ph, fr, ct = rng.uniform(0, 6.28), rng.uniform(.8, 1.2), rng.uniform(.6, 1.)
    if cls == FLOOR:
        base = .5 + .15 * np.sin(2 * np.pi * (rng.uniform(.3, .8) * yy + ph) * .5) * ct
    elif cls == WALL:                                    # brick-like horizontal courses with offset joints
        rows = (yy * 4 * fr + ph / 6.28).astype(int)
        base = .5 + .3 * ct * (((yy * 4 * fr + ph / 6.28) % 1 < .12) + ((xx * 4 + .5 * rows + ph) % 1 < .1)) - .15
    else:                                                # diagonal warning stripes
        base = .5 + .3 * ct * np.sign(np.sin(2 * np.pi * (xx + yy) * 3 * fr + ph))
    return np.clip(base + noise * rng.standard_normal((S, S)), 0, 1).astype(np.float32)


def labelled(n_per, rng, noise):
    y = np.repeat(np.arange(3), n_per); rng.shuffle(y)
    return np.stack([render(c, rng, noise) for c in y]), y


def make_task_map(rng):
    while True:
        g, s, goal, _ = make_map(rng, density=.16)
        haz = (rng.random((N, N)) < .07) & ~g; haz[s] = haz[goal] = False
        d = bfs(g | haz, goal)
        if d[s] >= 0 and abs(np.array(s) - np.array(goal)).sum() >= 12:
            return g, haz, s, goal, d


def navigate(g, haz, perceived, s, goal, d, w_mark=-1.3, decay=.9):
    """Fly using perceived classes. Returns 1 on success, 0 otherwise."""
    pos, marks, solid = s, np.zeros((N, N)), perceived.copy(); seen = {}
    for step in range(4 * int(d[s]) + 30):
        if pos == goal: return 1
        marks *= decay; marks[pos] += 1.
        best, bs = None, -1e18
        for dr, dc in MOVES:
            y, x = pos[0] + dr, pos[1] + dc
            if 0 <= y < N and 0 <= x < N and solid[y, x] == FLOOR:
                prog = (abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])) - (abs(goal[0] - y) + abs(goal[1] - x))
                sc = prog + w_mark * marks[y, x]
                if sc > bs: bs, best = sc, (y, x)
        if best is None: return 0
        if haz[best]: return 0                           # flew into a hazard
        if g[best]:                                      # bump: learns the cell is solid
            solid[best] = WALL; continue
        pos = best; seen[pos] = seen.get(pos, 0) + 1
        if seen[pos] > 8: return 0
    return 0


def truth_grid(g, haz):
    return np.where(g, WALL, np.where(haz, HAZARD, FLOOR))


class Mycelia:
    name = 'mycelia'
    def __init__(self, unlabeled, n_receptors):
        self.field = ReceptorField(ReceptorConfig(receptors=n_receptors, patch=6, grid=2, patches_per_image=12, iterations=10)).fit(list(unlabeled))
    def fit(self, X, y, seed):
        self.mem = TraceMemory(3, gamma=4., lam=.01).fit(self.field.encode_many(list(X)), y); return self
    def predict(self, X): return self.mem.predict(self.field.encode_many(list(X)))
    def size(self): return int(self.field.receptors.size + self.mem.traces.size)


class RawSVM:
    name = 'raw_svm'
    def fit(self, X, y, seed):
        self.m = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale')).fit(X.reshape(len(X), -1), y); self.n = self.m[-1].support_vectors_.size; return self
    def predict(self, X): return self.m.predict(X.reshape(len(X), -1))
    def size(self): return int(self.n)


class SmallCNN:
    name = 'cnn'
    def fit(self, X, y, seed):
        import torch, torch.nn as nn
        torch.manual_seed(seed); torch.set_num_threads(2); self.torch = torch
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16 * 16, 32), nn.ReLU(), nn.Linear(32, 3))
        opt = torch.optim.Adam(self.net.parameters(), 3e-3); Xt = torch.tensor(X[:, None]); yt = torch.tensor(y)
        n = len(X); rng = np.random.default_rng(seed)
        for _ in range(120):                              # fixed budget; no test data used
            idx = torch.tensor(rng.permutation(n)[:64]); opt.zero_grad()
            nn.functional.cross_entropy(self.net(Xt[idx]), yt[idx]).backward(); opt.step()
        return self
    def predict(self, X):
        with self.torch.no_grad(): return self.net(self.torch.tensor(X[:, None])).argmax(1).numpy()
    def size(self): return int(sum(p.numel() for p in self.net.parameters()))


def run(a):
    t0 = time.time(); res = []
    for noise in a.noise:
        rng = np.random.default_rng(1)
        unlabeled, _ = labelled(a.unlabeled // 3, rng, noise)
        myc_field = Mycelia(unlabeled, a.receptors)                    # label-free stage, once per noise level
        maps = [make_task_map(np.random.default_rng(777000 + i)) for i in range(a.maps)]
        cells_img = []; cells_true = []
        for k, (g, haz, s, goal, d) in enumerate(maps):
            t = truth_grid(g, haz); r = np.random.default_rng(31000 + k)
            cells_img.append(np.stack([render(c, r, noise) for c in t.ravel()])); cells_true.append(t)
        for n_per in a.labels:
            for seed in range(a.seeds):
                Xl, yl = labelled(n_per, np.random.default_rng(100 * seed + n_per), noise)
                for method in (myc_field, SmallCNN(), RawSVM()):
                    method.fit(Xl, yl, seed)
                    t1 = time.perf_counter(); pred = [method.predict(c).reshape(N, N) for c in cells_img]; dt = time.perf_counter() - t1
                    acc = float(np.mean([(p == t).mean() for p, t in zip(pred, cells_true)]))
                    haz_missed = float(np.mean([((p == FLOOR) & (t == HAZARD)).sum() / max((t == HAZARD).sum(), 1) for p, t in zip(pred, cells_true)]))
                    succ = float(np.mean([navigate(g, haz, p.copy(), s, goal, d) for p, (g, haz, s, goal, d) in zip(pred, maps)]))
                    row = {'noise': noise, 'labels_per_class': n_per, 'seed': seed, 'method': method.name, 'perception_acc': acc,
                           'hazard_missed': haz_missed, 'task_success': succ, 'ms_per_cell': 1000 * dt / (len(maps) * N * N), 'size_numbers': method.size()}
                    res.append(row)
                    print(f"noise {noise} labels/class {n_per:3d} seed {seed} {method.name:8s} acc {acc:.3f} hazard-missed {haz_missed:.3f} "
                          f"success {succ:.3f} {row['ms_per_cell']:.3f} ms/cell {time.time()-t0:.0f}s", flush=True)
        truth = float(np.mean([navigate(g, haz, truth_grid(g, haz), s, goal, d) for g, haz, s, goal, d in maps]))
        res.append({'noise': noise, 'method': 'ground_truth', 'task_success': truth}); print('noise', noise, 'ground-truth ceiling', truth, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--noise', type=float, nargs='+', default=[.1, .3, .5]); ap.add_argument('--labels', type=int, nargs='+', default=[5, 20, 100])
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--maps', type=int, default=40)
    ap.add_argument('--unlabeled', type=int, default=600); ap.add_argument('--receptors', type=int, default=120)
    ap.add_argument('--output', type=Path, default=Path('runs/swarm-vision'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    res = run(a); (a.output / 'results.json').write_text(json.dumps({'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, 'rows': res}, indent=2))
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        if r['method'] != 'ground_truth': agg[(r['noise'], r['labels_per_class'], r['method'])].append(r)
    print('\nSUMMARY (mean over seeds)  noise  labels  method  acc  success  ms/cell')
    for k, v in sorted(agg.items()):
        print(k, round(np.mean([x['perception_acc'] for x in v]), 3), round(np.mean([x['task_success'] for x in v]), 3), round(np.mean([x['ms_per_cell'] for x in v]), 3))


if __name__ == '__main__':
    main()
