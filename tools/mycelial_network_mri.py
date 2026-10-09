"""Brain-tumor MRI on the real mycelial network topology, with ablations.

Variants (all teach-only TraceMemory with mirror imprints, same receptors):
  isolated   - compartments sense their region, no signal spread
  uniform    - spread along untrained cords (all equal)
  learned    - spread along cords thickened by unsupervised Hebbian learning
Selection (variant, gamma) uses a 20% validation split of Training only.
Every variant is then scored on Testing and reported as an ablation; the
validation-selected one is the headline result.
"""
from __future__ import annotations
import argparse, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from PIL import Image
from mycelia.receptor_memory import ReceptorConfig, ReceptorField, TraceMemory
from mycelia.mycelial_network import MycelialNetwork, NetworkConfig
from train_mycelium import LABELS, load_images, clean_splits, write_json
from receptor_memory_mri import diagnosis_metrics, presence_metrics


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
    y = np.array([LABELS.index(r['label']) for r in records]); ytr = y[tr]
    field = ReceptorField(ReceptorConfig()).fit([X[i] for i in tr])
    print('receptors grown (unsupervised)', flush=True)
    base = MycelialNetwork(field, NetworkConfig())
    cache = a.output / 'cache'; cache.mkdir(exist_ok=True)
    def sensed(name, imgs):
        path = cache / f'{name}.npy'
        if not path.exists():
            first = base.sense(imgs[0]); out = np.lib.format.open_memmap(path, 'w+', np.float32, (len(imgs),) + first.shape)
            for k, img in enumerate(imgs): out[k] = base.sense(img)
            out.flush(); del out
        print('sensed', name, flush=True)
        return np.load(path, mmap_mode='r')
    S_tr = sensed('train', [X[i] for i in tr]); S_flip = sensed('flip', [X[i][:, ::-1].copy() for i in tr])
    S_te = sensed('test', [X[i] for i in te])

    nets = {'isolated': MycelialNetwork(field, NetworkConfig(steps=0)),
            'uniform': MycelialNetwork(field, NetworkConfig(learn_topology=False)),
            'learned': MycelialNetwork(field, NetworkConfig())}
    nets['learned'].learn_cords(np.asarray(S_tr[:, :, :]).mean(2, keepdims=True))  # unlabelled Training responses only
    rng = np.random.default_rng(a.seed)
    val = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:int(.2 * (ytr == c).sum())] for c in range(4)])
    inner = np.setdiff1d(np.arange(len(tr)), val)
    grid, results = [], {}
    for name, net in nets.items():
        F_tr, F_flip, F_te = net.settle(S_tr), net.settle(S_flip), net.settle(S_te)
        for gamma in (2., 5.):
            mem = TraceMemory(4, gamma, .003).fit(F_tr[inner], ytr[inner], [(F_flip[inner], ytr[inner])])
            acc = float((mem.predict(F_tr[val]) == ytr[val]).mean())
            grid.append({'variant': name, 'gamma': gamma, 'validation_accuracy': acc}); print(grid[-1], flush=True)
        g = max((r for r in grid if r['variant'] == name), key=lambda r: r['validation_accuracy'])['gamma']
        pred = TraceMemory(4, g, .003).fit(F_tr, ytr, [(F_flip, ytr)]).predict(F_te)
        results[name] = {'gamma': g, 'network': net.summary(), 'heldout_diagnosis': diagnosis_metrics(y[te], pred),
                         'heldout_tumor_presence': presence_metrics(y[te], pred)}
        print(name, 'test', results[name]['heldout_diagnosis']['accuracy'], flush=True)
        del F_tr, F_flip, F_te
    best = max(grid, key=lambda r: r['validation_accuracy'])
    write_json(a.output / 'mycelial-network-results.json', {
        'method': __doc__, 'receptor_config': asdict(ReceptorConfig()), 'network_config': asdict(NetworkConfig()),
        'clean_counts': {'Training': len(tr), 'Testing': len(te)}, 'validation_grid': grid,
        'selected_by_validation': best, 'headline': results[best['variant']], 'ablations': results,
        'seconds': time.monotonic() - start})
    h = results[best['variant']]
    print(f"selected {best['variant']}: diagnosis {h['heldout_diagnosis']['accuracy']:.4f}, "
          f"tumor presence {h['heldout_tumor_presence']['accuracy']:.4f}", flush=True)


if __name__ == '__main__':
    main()
