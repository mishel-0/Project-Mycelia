"""Frozen benchmark tiers and the multi-dimension scorecard.

`prepare` splits data once into search (inner Training), selection
(validation part of Training) and holdout (Testing). The lab can only open
`lab.npz` (search + selection); `holdout.npz` is read solely by the gate.
Hashes in the manifest make any change to the frozen data detectable.
"""
from __future__ import annotations
import hashlib, json, math, time
from dataclasses import fields, replace
from pathlib import Path
import numpy as np
from mycelia.symbiosis import SymbiosisEngine, SymbiosisConfig

LABELS = ('glioma', 'meningioma', 'notumor', 'pituitary')
LAB_TIERS = ('search', 'selection')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(features, labels, train_idx, test_idx, root, seed=1, val_fraction=.2):
    """features: dict tr/fl/te (n x compartments x dims); labels: per-record labels."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    ytr, yte = labels[train_idx], labels[test_idx]
    rng = np.random.default_rng(seed)
    val = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:int(val_fraction * (ytr == c).sum())] for c in range(4)])
    inner = np.setdiff1d(np.arange(len(ytr)), val)
    np.savez(root / 'lab.npz', search_x=features['tr'][inner], search_flip=features['fl'][inner], search_y=ytr[inner],
             selection_x=features['tr'][val], selection_y=ytr[val],
             train_x=features['tr'], train_flip=features['fl'], train_y=ytr)
    np.savez(root / 'holdout.npz', x=features['te'], y=yte)
    manifest = {'created': time.time(), 'seed': seed, 'labels': LABELS,
                'counts': {'search': len(inner), 'selection': len(val), 'holdout': len(yte)},
                'sha256': {'lab.npz': sha256(root / 'lab.npz'), 'holdout.npz': sha256(root / 'holdout.npz')}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


class LabBenchmark:
    """Search/selection tiers only. Opening holdout is not possible from here."""
    def __init__(self, root):
        self.root = Path(root); self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if sha256(self.root / 'lab.npz') != self.manifest['sha256']['lab.npz']:
            raise RuntimeError('frozen benchmark changed: lab.npz hash mismatch')
        d = np.load(self.root / 'lab.npz')
        self.tiers = {'search': (d['search_x'], d['search_flip'], d['search_y']),
                      'selection': (d['selection_x'], None, d['selection_y'])}

    def tier(self, name):
        if name not in LAB_TIERS:
            raise PermissionError(f'tier {name!r} is not available to the lab')
        return self.tiers[name]


def engine_config(candidate, seed):
    known = {f.name for f in fields(SymbiosisConfig)}; cfg = dict(candidate.get('engine', {}))
    if set(cfg) - known:
        raise ValueError(f'unknown engine settings: {sorted(set(cfg) - known)}')
    if 'label_margins' in cfg:
        cfg['label_margins'] = tuple(tuple(p) for p in cfg['label_margins'])
    return replace(SymbiosisConfig(**cfg), seed=seed)


def train(candidate, x, flip, y, seed):
    if candidate.get('mirror', True) and flip is not None:
        x = np.concatenate([x, flip]); y = np.concatenate([y, y])
    e = SymbiosisEngine(x.shape[1] * x.shape[2], 4, x.shape[1], engine_config(candidate, seed))
    return e.fit_stream(x.reshape(len(x), -1), y, int(candidate.get('epochs', 2)))


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [0., 1.]
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def evaluate(engine, x, y, coverage=.9):
    s = engine.scores(x.reshape(len(x), -1)).astype(np.float64)
    p = s.argmax(1); share = s / np.maximum(s.sum(1, keepdims=True), 1e-30)
    top = np.sort(share, 1); margin = top[:, -1] - top[:, -2]
    keep = margin >= np.quantile(margin, 1 - coverage) if coverage < 1 else np.ones(len(y), bool)
    recall = {LABELS[c]: float((p[y == c] == c).mean()) for c in range(4)}
    counts = {LABELS[c]: int((y == c).sum()) for c in range(4)}
    return {'accuracy': float((p == y).mean()), 'correct': int((p == y).sum()), 'n': len(y),
            'balanced_accuracy': float(np.mean(list(recall.values()))), 'recall': recall, 'class_counts': counts,
            'selective_accuracy': float((p[keep] == y[keep]).mean()), 'coverage': float(keep.mean()),
            'tumor_presence_accuracy': float(((p != 2) == (y != 2)).mean())}


def scorecard(candidate, bench, seeds=(0, 1, 2), robustness=True):
    """Train on search, score on selection, over several seeds."""
    sx, sf, sy = bench.tier('search'); vx, _, vy = bench.tier('selection'); runs = []
    for seed in seeds:
        t = time.monotonic(); e = train(candidate, sx, sf, sy, seed); fit_s = time.monotonic() - t
        r = evaluate(e, vx, vy); r.update(memories=int(e.n), fit_seconds=fit_s)
        if robustness:
            e.damage_traces(.3, np.random.default_rng(100 + seed))
            heal_x = np.concatenate([sx, sf]) if sf is not None else sx
            e.heal(heal_x.reshape(len(heal_x), -1), np.concatenate([sy, sy]) if sf is not None else sy, 1)
            r['robust_accuracy'] = evaluate(e, vx, vy)['accuracy']
        runs.append(r)
    keys = ['accuracy', 'balanced_accuracy', 'selective_accuracy', 'tumor_presence_accuracy', 'memories', 'fit_seconds'] + \
           (['robust_accuracy'] if robustness else [])
    summary = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    summary.update({k + '_sd': float(np.std([r[k] for r in runs])) for k in ('accuracy',)})
    summary['recall'] = {l: float(np.mean([r['recall'][l] for r in runs])) for l in LABELS}
    summary['recall_ci'] = {l: wilson(int(round(summary['recall'][l] * runs[0]['class_counts'][l])), runs[0]['class_counts'][l])
                            for l in LABELS}
    return {'seeds': list(seeds), 'summary': summary, 'runs': runs}
