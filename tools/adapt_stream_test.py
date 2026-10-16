"""Does MYCELIA adapt itself to a new environment without retraining?

A robot learns the textures of building A (a few labels), then is moved to
building B where everything looks different (rotated, lower contrast, darker,
noisier). In B it receives a stream of batches: many UNLABELLED camera tiles
and only a few labelled ones (feedback from touching/bumping). After every
batch we measure accuracy on held-out B tiles (adaptation) and on A tiles
(forgetting), plus the compute spent on the update.

Methods (same stream, same labels):
  frozen_mycelia    : never updates
  mycelia_memory    : adds labelled B tiles to its trace memory only
  mycelia_adapt     : also nudges its receptors with unlabelled B tiles
  frozen_cnn        : never updates
  cnn_finetune      : a few gradient steps per batch on a replay buffer
  cnn_retrain       : retrains from scratch on all labelled data each batch
Research simulation only.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from swarm_vision_test import render, labelled, Mycelia
from swarm_vision_pod import TunedCNN
from mycelia.receptor_memory import TraceMemory, _norm

METHODS = ('frozen_mycelia', 'mycelia_memory', 'mycelia_adapt', 'frozen_cnn', 'cnn_finetune', 'cnn_retrain')


def shift_img(img, rng, extra=.08):
    """Building B: rotated textures, lower contrast, darker, noisier."""
    return np.clip(.7 * (img.T - .5) + .5 - .08 + extra * rng.standard_normal(img.shape), 0, 1).astype(np.float32)


def make_B(n_per, rng):
    y = np.repeat(np.arange(3), n_per); rng.shuffle(y)
    return np.stack([shift_img(render(c, rng, .1), rng) for c in y]), y


def pick_labelled(X, y, k, rng):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in range(3)]); return X[idx], y[idx]


class AdaptiveMycelia:
    def __init__(self, base, adapt_receptors, cap=60):
        self.field, self.adapt, self.cap = base.field, adapt_receptors, cap
        self.bufX, self.bufy = {c: [] for c in range(3)}, {c: [] for c in range(3)}

    def add(self, X, y):
        for x, c in zip(X, y):
            self.bufX[int(c)].append(x); self.bufy[int(c)].append(int(c))
            if len(self.bufX[int(c)]) > self.cap: self.bufX[int(c)].pop(0); self.bufy[int(c)].pop(0)

    def refit(self):
        X = [x for c in range(3) for x in self.bufX[c]]; y = np.array([c for c in range(3) for _ in self.bufX[c]])
        self.mem = TraceMemory(3, gamma=4., lam=.01).fit(self.field.encode_many(X), y)

    def nudge_receptors(self, imgs, rng, lr=.03, per_image=15):
        f = self.field; c = f.config
        for img in imgs:
            p = f._patches(img).reshape(-1, c.patch * c.patch); p = p[rng.choice(len(p), per_image, replace=False)]
            z = (_norm(p) - f.mu) @ f.whiten; win = np.argmax(z @ f.receptors.T, 1)
            for zi, k in zip(z, win):
                f.receptors[k] += lr * zi / (np.linalg.norm(zi) + 1e-9); f.receptors[k] /= np.linalg.norm(f.receptors[k]) + 1e-9

    def update(self, Xlab, ylab, Xunl, rng):
        self.add(Xlab, ylab)
        if self.adapt: self.nudge_receptors(Xunl, rng)
        self.refit()

    def predict(self, X): return self.mem.predict(self.field.encode_many(list(X)))


def run_seed(seed, a):
    rng = np.random.default_rng(seed)
    unl, _ = labelled(a.unlabeled // 3, rng, .1); XA, yA = labelled(a.labels_A, rng, .1)
    A_test, yA_test = labelled(a.test // 3, np.random.default_rng(10_000 + seed), .1)
    B_test, yB_test = make_B(a.test // 3, np.random.default_rng(20_000 + seed))
    base_field = Mycelia(unl, a.receptors)
    methods = {}
    for name, ad in (('frozen_mycelia', None), ('mycelia_memory', False), ('mycelia_adapt', True)):
        m = AdaptiveMycelia(Mycelia(unl, a.receptors) if ad is not None else base_field, bool(ad))
        m.field = Mycelia(unl, a.receptors).field if False else m.field
        m.add(XA, yA); m.refit(); methods[name] = m
    # independent receptor copies so adapt/memory/frozen do not share state
    import copy
    for name in ('mycelia_memory', 'mycelia_adapt'):
        methods[name].field = copy.deepcopy(base_field.field); methods[name].refit()
    methods['frozen_mycelia'].field = copy.deepcopy(base_field.field); methods['frozen_mycelia'].refit()
    import torch, torch.nn as nn
    torch.set_num_threads(2)
    cnn = TunedCNN(.1); net0, _ = cnn._train(XA, yA, 200, seed)
    nets = {'frozen_cnn': net0, 'cnn_finetune': copy.deepcopy(net0)}
    replayX, replayY = list(XA), list(yA); opt = torch.optim.AdamW(nets['cnn_finetune'].parameters(), 1e-3, weight_decay=1e-3)
    labB_X, labB_y = [], []

    def cnn_acc(net, X, y):
        net.eval()
        with torch.no_grad(): return float((net(torch.tensor(np.asarray(X)[:, None])).argmax(1).numpy() == y).mean())

    def snapshot(t, secs):
        row = {'t': t}
        for name, m in methods.items():
            row[name] = {'B': float((m.predict(B_test) == yB_test).mean()), 'A': float((m.predict(A_test) == yA_test).mean()), 'secs': secs.get(name, 0.)}
        for name in ('frozen_cnn', 'cnn_finetune', 'cnn_retrain'):
            net = nets.get(name, nets.get('cnn_retrain_net'))
            if net is not None: row[name] = {'B': cnn_acc(net, B_test, yB_test), 'A': cnn_acc(net, A_test, yA_test), 'secs': secs.get(name, 0.)}
        return row
    nets['cnn_retrain_net'] = copy.deepcopy(net0)
    curve = [snapshot(0, {})]
    for t in range(1, a.batches + 1):
        XB, yB = make_B(a.stream_batch // 3, rng); Xl, yl = pick_labelled(XB, yB, a.labels_per_batch, rng)
        labB_X += list(Xl); labB_y += list(yl); secs = {}
        for name, m in methods.items():
            if name == 'frozen_mycelia': continue
            t0 = time.perf_counter(); m.update(Xl, yl, XB, np.random.default_rng(seed * 100 + t)); secs[name] = time.perf_counter() - t0
        t0 = time.perf_counter(); replayX += list(Xl); replayY += list(yl); net = nets['cnn_finetune']; net.train(); g = np.random.default_rng(seed + t)
        Xr, yr = torch.tensor(np.asarray(replayX)[:, None]), torch.tensor(np.asarray(replayY))
        for _ in range(a.finetune_steps):
            idx = torch.tensor(g.permutation(len(replayX))[:64]); opt.zero_grad(); nn.functional.cross_entropy(net(Xr[idx]), yr[idx]).backward(); opt.step()
        secs['cnn_finetune'] = time.perf_counter() - t0
        t0 = time.perf_counter(); Xall = np.asarray(list(XA) + labB_X); yall = np.asarray(list(yA) + labB_y)
        nets['cnn_retrain_net'], _ = cnn._train(Xall, yall, a.retrain_steps, seed + t); secs['cnn_retrain'] = time.perf_counter() - t0
        curve.append(snapshot(t, secs))
    return curve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--batches', type=int, default=12)
    ap.add_argument('--labels-A', dest='labels_A', type=int, default=15); ap.add_argument('--labels-per-batch', type=int, default=2)
    ap.add_argument('--stream-batch', type=int, default=60); ap.add_argument('--unlabeled', type=int, default=450)
    ap.add_argument('--receptors', type=int, default=100); ap.add_argument('--test', type=int, default=450)
    ap.add_argument('--finetune-steps', type=int, default=25); ap.add_argument('--retrain-steps', type=int, default=150)
    ap.add_argument('--output', type=Path, default=Path('runs/adapt-stream'))
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.time(); curves = []
    for s in range(a.seeds):
        curves.append(run_seed(s, a)); c = curves[-1]
        print(f"seed {s}: B acc t=0 -> t={a.batches}: " + ' '.join(f"{n}={c[0][n]['B']:.2f}->{c[-1][n]['B']:.2f}" for n in METHODS if n in c[0]) + f" {time.time()-t0:.0f}s", flush=True)
    names = [n for n in METHODS if n in curves[0][0]]
    mean = lambda n, key, t: float(np.mean([c[t][n][key] for c in curves]))
    summary = {n: {'B_curve': [mean(n, 'B', t) for t in range(a.batches + 1)], 'A_final': mean(n, 'A', a.batches),
                   'A_start': mean(n, 'A', 0), 'secs_per_update': float(np.mean([c[t][n]['secs'] for c in curves for t in range(1, a.batches + 1)]))} for n in names}
    rng = np.random.default_rng(0)
    def diff_ci(n1, n2):
        d = np.array([np.mean([cu[t][n1]['B'] - cu[t][n2]['B'] for t in range(1, a.batches + 1)]) for cu in curves])
        b = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        return {'mean_B_diff_over_stream': float(d.mean()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    summary['compare'] = {'adapt_vs_memory': diff_ci('mycelia_adapt', 'mycelia_memory'), 'adapt_vs_cnn_finetune': diff_ci('mycelia_adapt', 'cnn_finetune'),
                          'adapt_vs_cnn_retrain': diff_ci('mycelia_adapt', 'cnn_retrain'), 'adapt_vs_frozen_mycelia': diff_ci('mycelia_adapt', 'frozen_mycelia')}
    (a.output / 'results.json').write_text(json.dumps({'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, 'summary': summary, 'curves': curves}, indent=2))
    ts = sorted({t for t in (0, 1, 2, 4, 8, a.batches) if t <= a.batches})
    print('\nB-accuracy by batch (mean over seeds): t=' + ' '.join(f'{t:5d}' for t in ts))
    for n in names:
        print(f"{n:15s}   " + ' '.join(f"{summary[n]['B_curve'][t]:5.2f}" for t in ts) +
              f"   A kept {summary[n]['A_start']:.2f}->{summary[n]['A_final']:.2f}   {1000*summary[n]['secs_per_update']:.0f} ms/update")
    for k, v in summary['compare'].items(): print(k, v)


if __name__ == '__main__':
    main()
