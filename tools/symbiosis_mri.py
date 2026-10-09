"""Mycelial Symbiosis Engine on brain-tumor MRI: accuracy, budget, damage.

Input: sensed compartment responses cached by tools/mycelial_network_mri.py
(results/mycelial-network/cache). Each compartment compresses its own region
response (per-compartment PCA on unlabelled Training data).

Reports (Testing scored only after settings are chosen on validation):
  headline   - validation-selected engine vs the fixed kernel network
  ablations  - no plasticity / no local credit / no sleep
  budget     - Experiment C: accuracy vs maximum trace budget and recall energy
  damage     - Experiment A: kill 10/30/50% of compartments or traces;
               fixed network vs engine without and with self-healing
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from mycelia.symbiosis import SymbiosisEngine, SymbiosisConfig
from train_mycelium import LABELS, load_images, clean_splits, write_json
from receptor_memory_mri import diagnosis_metrics, presence_metrics


def compress(cache, k):
    a_tr = np.load(cache / 'train.npy', mmap_mode='r'); a_fl = np.load(cache / 'flip.npy', mmap_mode='r')
    a_te = np.load(cache / 'test.npy', mmap_mode='r'); out = {'tr': [], 'fl': [], 'te': []}
    for j in range(a_tr.shape[1]):
        x = np.array(a_tr[:, j], dtype=np.float32); m = x.mean(0); s = x.std(0) + 1e-6
        _, _, vt = np.linalg.svd(((x - m) / s)[::2], full_matrices=False); v = vt[:k].T
        for key, arr in (('tr', a_tr), ('fl', a_fl), ('te', a_te)):
            out[key].append(((np.array(arr[:, j], dtype=np.float32) - m) / s) @ v)
    return {key: np.stack(v, 1) for key, v in out.items()}


class FixedNetwork:
    """Conventional baseline: every image stored, weights from one solve, no plasticity."""
    def __init__(self, gamma=5., lam=.003): self.gamma, self.lam = gamma, lam
    def _n(self, x, alive):
        x = np.array(x, dtype=np.float32); x[:, ~alive] = 0; x = x.reshape(len(x), -1)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    def fit(self, x, y):
        self.alive = np.ones(x.shape[1], dtype=bool); self.raw = x; self.t = self._n(x, self.alive)
        k = np.exp(self.gamma * (self.t @ self.t.T - 1)); k[np.diag_indices_from(k)] += self.lam
        self.a = np.linalg.solve(k, np.eye(4)[y] * 2 - 1); return self
    def predict(self, x):
        t = self._n(self.raw, self.alive)  # damage removes signal from stored memory too; weights stay fixed
        return (np.exp(self.gamma * (self._n(x, self.alive) @ t.T - 1)) @ self.a).argmax(1)


def run_engine(cfg, x, y, epochs, comps):
    return SymbiosisEngine(x.shape[1] * x.shape[2], 4, comps, cfg).fit_stream(x.reshape(len(x), -1), y, epochs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('results/mycelial-network/cache'))
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--dims', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    records, _ = load_images(a.dataset, 16); clean, _ = clean_splits(records)
    y = np.array([LABELS.index(r['label']) for r in records]); ytr, yte = y[clean['Training']], y[clean['Testing']]
    P = compress(a.cache, a.dims); comps = P['tr'].shape[1]
    X = np.concatenate([P['tr'], P['fl']]); Y = np.concatenate([ytr, ytr]); n = len(ytr); T = P['te']
    print(f'compressed to {comps} compartments x {a.dims}', flush=True)

    rng = np.random.default_rng(a.seed)
    val = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:int(.2 * (ytr == c).sum())] for c in range(4)])
    inner = np.setdiff1d(np.arange(n), val); idx = np.concatenate([inner, inner + n])
    grid = []
    for gamma in (10., 20., 30.):
        for margin in (.2, .5):
            for epochs in (1, 2):
                cfg = SymbiosisConfig(gamma=gamma, grow_margin=margin)
                eng = run_engine(cfg, X[idx], Y[idx], epochs, comps)
                acc = float((eng.predict(X[val].reshape(len(val), -1)) == ytr[val]).mean())
                grid.append({'gamma': gamma, 'grow_margin': margin, 'epochs': epochs, 'validation_accuracy': acc,
                             'traces': eng.summary()['traces']}); print(grid[-1], flush=True)
    best = max(grid, key=lambda g: (g['validation_accuracy'], -g['traces']))
    cfg = SymbiosisConfig(gamma=best['gamma'], grow_margin=best['grow_margin']); epochs = best['epochs']
    Tf = T.reshape(len(T), -1)

    def evaluate(eng):
        p = eng.predict(Tf if isinstance(eng, SymbiosisEngine) else T)
        return {'accuracy': float((p == yte).mean()), 'diagnosis': diagnosis_metrics(yte, p),
                'tumor_presence': presence_metrics(yte, p)}

    engine = run_engine(cfg, X, Y, epochs, comps); fixed = FixedNetwork().fit(X, Y)
    head = {'engine': {**evaluate(engine), 'memory': engine.summary(), 'recall_cost_per_image': engine.n},
            'fixed_network': {**evaluate(fixed), 'recall_cost_per_image': len(X)}}
    print('engine test', head['engine']['accuracy'], 'traces', engine.n, '| fixed', head['fixed_network']['accuracy'], len(X), flush=True)

    ablations = {}
    for name, change in (('no_plasticity', {'plasticity': False}), ('no_local_credit', {'local_credit': False}),
                         ('no_sleep', {'sleep': False})):
        e = run_engine(replace(cfg, **change), X, Y, epochs, comps)
        r = evaluate(e); ablations[name] = {'accuracy': r['accuracy'], 'traces': e.n}; print(name, ablations[name], flush=True)

    budget = []
    for b in (100, 250, 500, 1000, 2000):
        e = run_engine(replace(cfg, budget=b), X, Y, epochs, comps)
        budget.append({'budget': b, 'traces': e.n, 'accuracy': evaluate(e)['accuracy'],
                       'training_energy': e.stats['energy'], 'recall_cost_per_image': e.n}); print(budget[-1], flush=True)

    damage = []
    for kind in ('compartments', 'traces'):
        for frac in (.1, .3, .5):
            rows = []
            for r in range(a.repeats):
                dr = np.random.default_rng(100 + r)
                e = run_engine(cfg, X, Y, epochs, comps)
                if kind == 'compartments':
                    dead = e.damage_compartments(frac, dr)
                    f = FixedNetwork().fit(X, Y); f.alive[dead] = False; fixed_acc = evaluate(f)['accuracy']
                else:
                    e.damage_traces(frac, dr)
                    f = FixedNetwork().fit(X, Y); keep = np.ones(len(X), dtype=bool)
                    keep[dr.choice(len(X), int(round(frac * len(X))), replace=False)] = False
                    f.raw, f.a = f.raw[keep], f.a[keep]; fixed_acc = evaluate(f)['accuracy']
                damaged = evaluate(e)['accuracy']
                e.heal(X.reshape(len(X), -1), Y, 1); healed = evaluate(e)['accuracy']
                rows.append((fixed_acc, damaged, healed))
            m = np.mean(rows, 0)
            damage.append({'damage': kind, 'fraction': frac, 'fixed_network': float(m[0]),
                           'engine_damaged': float(m[1]), 'engine_self_healed': float(m[2]), 'repeats': a.repeats})
            print(damage[-1], flush=True)

    write_json(a.output / 'symbiosis-results.json', {
        'method': __doc__, 'config': asdict(cfg), 'epochs': epochs, 'compressed_dims_per_compartment': a.dims,
        'validation_grid': grid, 'selected': best, 'headline': head, 'ablations': ablations,
        'budget_experiment': budget, 'damage_experiment': damage, 'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
