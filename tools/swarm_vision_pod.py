"""Pod-scale swarm-vision benchmark: fair CNN baselines + significance tests.

Extends tools/swarm_vision_test.py. Differences that matter for a fair fight:
  * cnn_tuned: wider CNN, data augmentation (flips, shifts, extra noise) and a
    training length chosen on a held-out slice of the labelled set (never on
    test data). cnn_basic and raw_svm are kept as references.
  * many more maps/seeds, label budgets from 5 to 150 per class, three noise levels.
  * mycelia vs cnn_tuned compared with a paired bootstrap over seeds
    (95% interval on the success-rate difference per noise/label cell).
  * a single-process speed benchmark (ms per cell, no CPU contention).
Runs configurations in parallel (one per worker). CPU only. Research simulation.
"""
from __future__ import annotations
import argparse, json, time, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker: no oversubscription
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from swarm_vision_test import (render, labelled, make_task_map, navigate, truth_grid, Mycelia, SmallCNN, RawSVM, N)

METHODS = ('mycelia', 'cnn_tuned', 'cnn_basic', 'raw_svm')


def basic_cnn():
    m = SmallCNN(); m.name = 'cnn_basic'; return m


class TunedCNN:
    name = 'cnn_tuned'
    def __init__(self, noise): self.noise = noise
    def _net(self):
        import torch.nn as nn
        return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(.2), nn.Linear(32 * 16, 64), nn.ReLU(), nn.Linear(64, 3))
    def _aug(self, x, g):
        import torch
        if g.random() < .5: x = x.flip(3)
        if g.random() < .5: x = x.flip(2)
        x = torch.roll(x, (int(g.integers(-2, 3)), int(g.integers(-2, 3))), (2, 3))
        return x + float(g.uniform(0, .8 * self.noise)) * torch.randn_like(x)
    def _train(self, X, y, steps, seed, Xv=None, yv=None, every=25):
        import torch, torch.nn as nn
        torch.manual_seed(seed); torch.set_num_threads(1); g = np.random.default_rng(seed)
        net = self._net(); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
        Xt, yt = torch.tensor(X[:, None]), torch.tensor(y); hist = []
        for step in range(1, steps + 1):
            net.train(); idx = torch.tensor(g.permutation(len(X))[:64]); opt.zero_grad()
            nn.functional.cross_entropy(net(self._aug(Xt[idx], g)), yt[idx]).backward(); opt.step()
            if Xv is not None and step % every == 0:
                net.eval()
                with torch.no_grad(): hist.append((step, float((net(torch.tensor(Xv[:, None])).argmax(1).numpy() == yv).mean())))
        return net, hist
    def fit(self, X, y, seed):
        import torch
        self.torch = torch; rng = np.random.default_rng(seed); order = rng.permutation(len(X)); cut = max(int(.8 * len(X)), 3)
        tr, va = order[:cut], order[cut:]
        _, hist = self._train(X[tr], y[tr], 400, seed, X[va], y[va])
        best_steps = max(hist, key=lambda h: (h[1], -h[0]))[0] if hist else 200      # earliest best step on validation
        self.net, _ = self._train(X, y, best_steps, seed); self.net.eval(); self.steps = best_steps
        return self
    def predict(self, X):
        with self.torch.no_grad(): return self.net(self.torch.tensor(X[:, None])).argmax(1).numpy()
    def size(self): return int(sum(p.numel() for p in self.net.parameters()))


def worker(job):
    noise, n_per, seed, a = job
    os.environ['OMP_NUM_THREADS'] = '1'
    rng = np.random.default_rng(1 + seed); unlabeled, _ = labelled(a['unlabeled'] // 3, rng, noise)
    myc = Mycelia(unlabeled, a['receptors'])
    maps = [make_task_map(np.random.default_rng(777000 + i)) for i in range(a['maps'])]
    imgs, truth = [], []
    for k, (g, haz, s, goal, d) in enumerate(maps):
        t = truth_grid(g, haz); r = np.random.default_rng(31000 + k)
        imgs.append(np.stack([render(c, r, noise) for c in t.ravel()])); truth.append(t)
    Xl, yl = labelled(n_per, np.random.default_rng(100 * seed + n_per), noise); rows = []
    for method in (myc, TunedCNN(noise), basic_cnn(), RawSVM()):
        method.fit(Xl, yl, seed)
        pred = [method.predict(c).reshape(N, N) for c in imgs]
        per_map = [navigate(g, haz, p.copy(), s, goal, d) for p, (g, haz, s, goal, d) in zip(pred, maps)]
        rows.append({'noise': noise, 'labels_per_class': n_per, 'seed': seed, 'method': method.name,
                     'perception_acc': float(np.mean([(p == t).mean() for p, t in zip(pred, truth)])),
                     'task_success': float(np.mean(per_map)), 'per_map_success': per_map})
    truth_succ = float(np.mean([navigate(g, haz, truth_grid(g, haz), s, goal, d) for g, haz, s, goal, d in maps]))
    for r in rows: r['ground_truth_ceiling'] = truth_succ
    return rows


def boot_ci(diffs, rng, n=2000):
    diffs = np.asarray(diffs); b = [rng.choice(diffs, len(diffs)).mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def speed_benchmark(receptors, noise=.15, cells=8000):
    rng = np.random.default_rng(0); unlabeled, _ = labelled(150, rng, noise); Xl, yl = labelled(40, rng, noise)
    Xt, _ = labelled(cells // 3, rng, noise); out = {}
    for m in (Mycelia(unlabeled, receptors), TunedCNN(noise), basic_cnn(), RawSVM()):
        m.fit(Xl, yl, 0); m.predict(Xt[:200]); t = time.perf_counter(); m.predict(Xt)
        out[m.name] = {'ms_per_cell': 1000 * (time.perf_counter() - t) / len(Xt), 'numbers_stored': m.size()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--noise', type=float, nargs='+', default=[.05, .15, .3])
    ap.add_argument('--labels', type=int, nargs='+', default=[5, 10, 20, 40, 80, 150])
    ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--maps', type=int, default=100)
    ap.add_argument('--unlabeled', type=int, default=900); ap.add_argument('--receptors', type=int, default=100)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--output', type=Path, default=Path('runs/swarm-vision-pod'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.time()
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    jobs = [(n, l, s, cfg) for n in a.noise for l in a.labels for s in range(a.seeds)]
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, jobs)):
            rows.extend(r)
            if i % 2 == 0: print(f'{i+1}/{len(jobs)} configs done, {time.time()-t0:.0f}s', flush=True)
    rng = np.random.default_rng(0); summary = {}
    for n in a.noise:
        for l in a.labels:
            cell = {m: [r['task_success'] for r in rows if r['noise'] == n and r['labels_per_class'] == l and r['method'] == m] for m in METHODS}
            acc = {m: [r['perception_acc'] for r in rows if r['noise'] == n and r['labels_per_class'] == l and r['method'] == m] for m in METHODS}
            seeds_sorted = {m: [r['task_success'] for r in sorted((r for r in rows if r['noise'] == n and r['labels_per_class'] == l and r['method'] == m), key=lambda r: r['seed'])] for m in METHODS}
            diff = np.array(seeds_sorted['mycelia']) - np.array(seeds_sorted['cnn_tuned'])
            summary[f'noise{n}_labels{l}'] = {'success': {m: float(np.mean(v)) for m, v in cell.items()},
                                              'perception_acc': {m: float(np.mean(v)) for m, v in acc.items()},
                                              'mycelia_minus_cnn_tuned': float(diff.mean()), 'diff_ci95': boot_ci(diff, rng),
                                              'mycelia_beats_cnn_tuned_significantly': bool(boot_ci(diff, rng)[0] > 0),
                                              'cnn_tuned_beats_mycelia_significantly': bool(boot_ci(diff, rng)[1] < 0)}
    speed = {str(r): speed_benchmark(r) for r in (50, a.receptors)}
    out = {'config': cfg, 'summary': summary, 'speed': speed, 'seconds': time.time() - t0,
           'rows': [{k: v for k, v in r.items() if k != 'per_map_success'} for r in rows]}
    (a.output / 'results.json').write_text(json.dumps(out, indent=2))
    print('RESULT_SUMMARY_BEGIN'); print(json.dumps({'summary': summary, 'speed': speed, 'seconds': out['seconds']})); print('RESULT_SUMMARY_END')


if __name__ == '__main__':
    main()
