"""Mycelia Collective Dissent: disagreement detection and abstention.

Specialists see the same scan through different evidence:
  A engine   - Symbiosis Engine on compartment responses (plastic memory)
  B fixed    - fixed kernel network on the same compartment responses
  C pixels   - kernel memory on raw 64px pixels (independent features)
  D failures - similarity to scans the engine got wrong in 2-fold
               cross-validation on training data ("known failure patterns")

Coordinator: logistic model predicting "A is wrong" from disagreement,
margins and failure similarity. It is fitted on a validation split of
Training only; abstention thresholds for target coverages are fixed there.
Specialists are then refit on all Training and Testing is scored once.
Abstention is a research safety signal, not clinical reliability.
"""
from __future__ import annotations
import argparse, math, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
from mycelia.symbiosis import SymbiosisEngine, SymbiosisConfig
from train_mycelium import LABELS, load_images, clean_splits, write_json
from symbiosis_mri import compress, FixedNetwork
from receptor_memory_mri import wilson

ENGINE = SymbiosisConfig(gamma=30., grow_margin=.5)  # selected on validation in symbiosis_mri
COVERAGES = (1., .95, .9, .85, .8, .7)


def softmax_margin(s):
    s = np.asarray(s, dtype=np.float64); e = np.exp(s - s.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
    top = np.sort(p, 1); return p, top[:, -1] - top[:, -2]


class PixelMemory:
    def __init__(self, gamma=10., lam=.01): self.gamma, self.lam = gamma, lam
    def _n(self, x):
        x = x.reshape(len(x), -1).astype(np.float32); x = x - x.mean(1, keepdims=True)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    def fit(self, x, y):
        self.t = self._n(x); k = np.exp(self.gamma * (self.t @ self.t.T - 1)); k[np.diag_indices_from(k)] += self.lam
        self.a = np.linalg.solve(k, np.eye(4)[y] * 2 - 1); return self
    def scores(self, x): return np.exp(self.gamma * (self._n(x) @ self.t.T - 1)) @ self.a


def engine(x, y, seed=0):
    return SymbiosisEngine(x.shape[1] * x.shape[2], 4, x.shape[1], replace(ENGINE, seed=seed)).fit_stream(
        x.reshape(len(x), -1), y, 2)


def failure_bank(x, y, seed):
    """Scans the engine misclassifies under 2-fold cross-validation (labels of Training only)."""
    rng = np.random.default_rng(seed); order = rng.permutation(len(x)); half = np.array_split(order, 2); bank = []
    for a, b in ((0, 1), (1, 0)):
        e = engine(x[half[a]], y[half[a]], seed)
        wrong = half[b][e.predict(x[half[b]].reshape(len(half[b]), -1)) != y[half[b]]]
        bank.append(wrong)
    bank = np.concatenate(bank); f = x[bank].reshape(len(bank), -1)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def signals(xc, xp, models, bank):
    eng, fixed, pix = models
    sa = eng.scores(xc.reshape(len(xc), -1)); sb = fixed_scores(fixed, xc); sc = pix.scores(xp)
    pa, ma = softmax_margin(np.log(np.maximum(sa, 1e-30))); pb, mb = softmax_margin(sb * 5); pc, mc = softmax_margin(sc * 5)
    a, b, c = pa.argmax(1), pb.argmax(1), pc.argmax(1)
    q = xc.reshape(len(xc), -1); q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    fail = (q @ bank.T).max(1) if len(bank) else np.zeros(len(q))
    distinct = np.array([len({i, j, k}) for i, j, k in zip(a, b, c)])
    feats = np.stack([ma, mb, mc, (a != b), (a != c), (b != c), distinct - 1, fail, pa.max(1)], 1).astype(np.float64)
    return a, feats, {'A': a, 'B': b, 'C': c}


def fixed_scores(f, x):
    t = f._n(f.raw, f.alive); return np.exp(f.gamma * (f._n(x, f.alive) @ t.T - 1)) @ f.a


def fit_logistic(x, t, l2=1e-2, iters=3000, lr=.5):
    m = x.mean(0); s = x.std(0) + 1e-9; z = (x - m) / s; w = np.zeros(z.shape[1]); b = 0.
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z @ w + b))); g = p - t
        w -= lr * (z.T @ g / len(t) + l2 * w); b -= lr * g.mean()
    return lambda q: 1 / (1 + np.exp(-(((q - m) / s) @ w + b))), dict(weights=w.tolist(), bias=float(b))


def risk_coverage(pred, y, risk, thresholds):
    rows = []
    for cov, th in thresholds:
        keep = risk <= th; k = int(keep.sum()); correct = int((pred[keep] == y[keep]).sum())
        g = y == LABELS.index('glioma'); kg = keep & g
        rows.append({'target_coverage': cov, 'threshold': float(th), 'coverage': k / len(y), 'answered': k,
                     'abstained': len(y) - k, 'accuracy_answered': correct / max(k, 1),
                     'accuracy_answered_wilson_95': wilson(correct, k) if k else None,
                     'errors_answered': k - correct, 'errors_caught': int(((pred != y) & ~keep).sum()),
                     'total_errors': int((pred != y).sum()),
                     'glioma_answered_recall': float((pred[kg] == y[kg]).mean()) if kg.sum() else None,
                     'glioma_abstained': int((g & ~keep).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('results/mycelial-network/cache'))
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    records, _ = load_images(a.dataset, 16); clean, _ = clean_splits(records); tr, te = clean['Training'], clean['Testing']
    y = np.array([LABELS.index(r['label']) for r in records]); ytr, yte = y[tr], y[te]
    def px(r, flip=False):
        with Image.open(a.dataset / r['file']) as im:
            g = np.asarray(im.convert('L').resize((64, 64), Image.Resampling.BOX), dtype=np.float32) / 255
        return g[:, ::-1] if flip else g
    Ptr = np.stack([px(records[i]) for i in tr]); Pfl = Ptr[:, :, ::-1]; Pte = np.stack([px(records[i]) for i in te])
    C = compress(a.cache, 16); Ctr, Cfl, Cte = C['tr'], C['fl'], C['te']; print('features ready', flush=True)

    def build(idx):
        xc = np.concatenate([Ctr[idx], Cfl[idx]]); xp = np.concatenate([Ptr[idx], Pfl[idx]]); yy = np.concatenate([ytr[idx], ytr[idx]])
        models = (engine(xc, yy), FixedNetwork().fit(xc, yy), PixelMemory().fit(xp, yy))
        return models, failure_bank(Ctr[idx], ytr[idx], a.seed)

    rng = np.random.default_rng(a.seed)
    val = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:int(.2 * (ytr == c).sum())] for c in range(4)])
    inner = np.setdiff1d(np.arange(len(tr)), val)
    models, bank = build(inner)
    pv, fv, specv = signals(Ctr[val], Ptr[val], models, bank); wrong_v = (pv != ytr[val]).astype(float)
    print('validation: engine errors', int(wrong_v.sum()), 'of', len(val), flush=True)
    names = ['margin_A', 'margin_B', 'margin_C', 'A!=B', 'A!=C', 'B!=C', 'disagreement', 'failure_similarity', 'confidence_A']
    coord, coef = fit_logistic(fv, wrong_v)
    rv = coord(fv); thresholds = [(c, np.quantile(rv, c) if c < 1 else 2.) for c in COVERAGES]
    val_curve = risk_coverage(pv, ytr[val], rv, thresholds)
    simple_v = (fv[:, 6] > 0).astype(float)  # disagreement-only rule
    print('coordinator fitted on validation', flush=True)

    models, bank = build(np.arange(len(tr)))
    pt, ft, spect = signals(Cte, Pte, models, bank); rt = coord(ft)
    test_curve = risk_coverage(pt, yte, rt, thresholds)
    dis = ft[:, 6] > 0; wrong_t = pt != yte
    disagreement = {'flagged': int(dis.sum()), 'errors_flagged': int((dis & wrong_t).sum()), 'total_errors': int(wrong_t.sum()),
                    'accuracy_when_all_agree': float((pt[~dis] == yte[~dis]).mean()), 'answered_when_all_agree': int((~dis).sum()),
                    'validation_flagged': int(simple_v.sum())}
    specialists = {k: float((v == yte).mean()) for k, v in spect.items()}
    write_json(a.output / 'collective-dissent-results.json', {
        'method': __doc__, 'specialist_test_accuracy': specialists, 'signals': names, 'coordinator': coef,
        'validation_curve': val_curve, 'test_curve': test_curve, 'disagreement_only_rule_test': disagreement,
        'failure_bank_size': int(len(bank)), 'seconds': time.monotonic() - start})
    print('specialists', specialists, flush=True); print('disagreement rule', disagreement, flush=True)
    for r in test_curve:
        print(f"target {r['target_coverage']:.2f}: coverage {r['coverage']:.3f}, accuracy {r['accuracy_answered']:.4f}, "
              f"errors caught {r['errors_caught']}/{r['total_errors']}, glioma recall {r['glioma_answered_recall']}", flush=True)


if __name__ == '__main__':
    main()
