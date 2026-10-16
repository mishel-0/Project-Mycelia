"""Brain-tumor MRI experiment: unsupervised receptor field + teach-only memory.

1. Unsupervised receptor learning on clean Training images only (no labels).
2. Settings (mirror imprints, gamma, lambda) chosen on a stratified 20%
   validation split of Training. Testing labels are never used for selection.
3. Refit memory on all clean Training (+ mirror imprints if selected) and score
   the held-out Testing split once: 4-class diagnosis and tumor presence.
Research prototype only; not clinically validated.
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
from mycelia.receptor_memory import ReceptorConfig, ReceptorField, TraceMemory
from train_mycelium import LABELS, load_images, clean_splits, write_json

NO_TUMOR = LABELS.index('notumor')


def wilson(k, n, z=1.959963984540054):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def diagnosis_metrics(y, p):
    m = np.zeros((4, 4), dtype=int)
    for a, b in zip(y, p): m[a, b] += 1
    recall = np.diag(m) / m.sum(1); precision = np.diag(m) / np.maximum(m.sum(0), 1)
    correct = int(np.trace(m))
    return {'accuracy': correct / len(y), 'correct': correct, 'n': len(y), 'wilson_95': wilson(correct, len(y)),
            'balanced_accuracy': float(recall.mean()), 'labels': list(LABELS), 'confusion_matrix': m.tolist(),
            'classes': {l: {'count': int(m[i].sum()), 'recall': float(recall[i]), 'precision': float(precision[i]),
                            'f1': float(2 * recall[i] * precision[i] / max(recall[i] + precision[i], 1e-12))}
                        for i, l in enumerate(LABELS)}}


def presence_metrics(y, p):
    t, g = y != NO_TUMOR, p != NO_TUMOR
    tp, fn, fp, tn = int((t & g).sum()), int((t & ~g).sum()), int((~t & g).sum()), int((~t & ~g).sum())
    n = len(y)
    return {'accuracy': (tp + tn) / n, 'wilson_95': wilson(tp + tn, n), 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1),
            'ppv': tp / max(tp + fp, 1), 'npv': tn / max(tn + fn, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--size', type=int, default=64); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    records, _ = load_images(a.dataset, 16); clean, audit = clean_splits(records)
    tr, te = clean['Training'], clean['Testing']
    def gray(r):
        with Image.open(a.dataset / r['file']) as im:
            return np.asarray(im.convert('L').resize((a.size, a.size), Image.Resampling.BOX), dtype=np.float32) / 255
    X = {i: gray(records[i]) for i in np.concatenate([tr, te])}
    y = np.array([LABELS.index(r['label']) for r in records])
    print(f'clean Training={len(tr)} Testing={len(te)}', flush=True)

    field = ReceptorField(ReceptorConfig()).fit([X[i] for i in tr])  # no labels
    print('receptor field grown (unsupervised)', flush=True)
    F_tr = field.encode_many([X[i] for i in tr]); F_te = field.encode_many([X[i] for i in te])
    F_flip = field.encode_many([X[i][:, ::-1].copy() for i in tr])
    print('encoded', flush=True)

    rng = np.random.default_rng(a.seed); ytr = y[tr]
    val = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:int(.2 * (ytr == c).sum())] for c in range(4)])
    inner = np.setdiff1d(np.arange(len(tr)), val); grid = []
    for mirror in (False, True):
        for gamma in (2., 5.):
            for lam in (.003, .01, .03):
                extra = [(F_flip[inner], ytr[inner])] if mirror else []
                mem = TraceMemory(4, gamma, lam).fit(F_tr[inner], ytr[inner], extra)
                acc = float((mem.predict(F_tr[val]) == ytr[val]).mean())
                grid.append({'mirror': mirror, 'gamma': gamma, 'lam': lam, 'validation_accuracy': acc})
                print(grid[-1], flush=True)
    best = max(grid, key=lambda g: g['validation_accuracy'])
    extra = [(F_flip, ytr)] if best['mirror'] else []
    mem = TraceMemory(4, best['gamma'], best['lam']).fit(F_tr, ytr, extra)
    pred = mem.predict(F_te)
    result = {'method': __doc__, 'receptor_config': asdict(ReceptorConfig()), 'image_size': a.size,
              'clean_counts': {'Training': len(tr), 'Testing': len(te)}, 'audit': audit,
              'validation_grid': grid, 'selected': best,
              'heldout_diagnosis': diagnosis_metrics(y[te], pred),
              'heldout_tumor_presence': presence_metrics(y[te], pred),
              'seconds': time.monotonic() - start,
              'limits': 'Image-level split from one public dataset; patient identity unknown and near-duplicate '
                        'images exist across splits. Research only; not a medical device.'}
    write_json(a.output / 'receptor-memory-results.json', result)
    d, t = result['heldout_diagnosis'], result['heldout_tumor_presence']
    print(f"diagnosis accuracy {d['accuracy']:.4f} ({d['correct']}/{d['n']}); tumor presence accuracy {t['accuracy']:.4f} "
          f"sensitivity {t['sensitivity']:.4f} specificity {t['specificity']:.4f}", flush=True)


if __name__ == '__main__':
    main()
