"""Build the saved Mycelia pipeline used by `python -m mycelia.talk`.

Same protocol as tools/mlcp_mri.py: specialists on inner Training build the
validation node states for the MLCP codebooks (K=16, L=1) and receiver; the
specialists are then refit on all Training. Requires the compartment cache
from tools/mycelial_network_mri.py and the receptor field it used (refit
deterministically here).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from mycelia.receptor_memory import ReceptorConfig, ReceptorField
from mycelia.mlcp.codebook import Codebook
from mycelia.mlcp.nodes import SENDERS, LABELS
from mycelia.mlcp.receiver import Receiver
from mycelia.talk.bundle import MyceliaBundle, load_gray
from collective_dissent_mri import engine, PixelMemory, failure_bank
from symbiosis_mri import FixedNetwork
from mlcp_mri import node_states, transmit
from train_mycelium import load_images, clean_splits


def compress_params(cache, k=16):
    a_tr = np.load(cache / 'train.npy', mmap_mode='r'); a_fl = np.load(cache / 'flip.npy', mmap_mode='r'); params, tr, fl = [], [], []
    for j in range(a_tr.shape[1]):
        x = np.array(a_tr[:, j], dtype=np.float32); m = x.mean(0); s = x.std(0) + 1e-6
        _, _, vt = np.linalg.svd(((x - m) / s)[::2], full_matrices=False); v = vt[:k].T
        params.append((m, s, v)); tr.append(((x - m) / s) @ v); fl.append(((np.array(a_fl[:, j], dtype=np.float32) - m) / s) @ v)
    return params, np.stack(tr, 1), np.stack(fl, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('results/mycelial-network/cache'))
    ap.add_argument('--output', type=Path, default=Path('lab_state/talk/mycelia-bundle.pkl'))
    ap.add_argument('--check', type=int, default=200, help='test images to re-diagnose end to end')
    a = ap.parse_args()
    records, _ = load_images(a.dataset, 16); clean, _ = clean_splits(records); tr, te = clean['Training'], clean['Testing']
    y = np.array([LABELS.index(r['label']) for r in records]); ytr = y[tr]
    G = {i: load_gray(a.dataset / records[i]['file']) for i in tr}
    field = ReceptorField(ReceptorConfig()).fit([G[i] for i in tr]); print('receptor field', flush=True)
    params, Ctr, Cfl = compress_params(a.cache); print('compression', flush=True)
    Ptr = np.stack([G[i] for i in tr]); Pfl = Ptr[:, :, ::-1]

    def build(idx):
        xc = np.concatenate([Ctr[idx], Cfl[idx]]); xp = np.concatenate([Ptr[idx], Pfl[idx]]); yy = np.concatenate([ytr[idx], ytr[idx]])
        return (engine(xc, yy), FixedNetwork().fit(xc, yy), PixelMemory().fit(xp, yy)), failure_bank(Ctr[idx], ytr[idx], 1)

    rng = np.random.default_rng(1)
    val = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:int(.2 * (ytr == c).sum())] for c in range(4)])
    inner = np.setdiff1d(np.arange(len(tr)), val)
    models, bank = build(inner); Sv = node_states(models, bank, Ctr[val], Ptr[val])
    books = {s: Codebook(16, 1, seed=1).fit(Sv[s]) for s in SENDERS}
    dec, miss, leans, _, _ = transmit(Sv, 'mlcp', books); receiver = Receiver().fit(dec, miss, ytr[val], leans)
    print('codebooks and receiver (validation)', flush=True)
    (eng, fixed, pix), bank = build(np.arange(len(tr)))
    bundle = MyceliaBundle(field=field, compress=params, engine=eng, bank=bank, books=books, receiver=receiver,
                           fixed={'t': fixed._n(fixed.raw, fixed.alive), 'a': fixed.a, 'gamma': fixed.gamma},
                           pixels={'t': pix.t, 'a': pix.a, 'gamma': pix.gamma},
                           meta={'labels': LABELS, 'training_images': int(len(tr)), 'protocol': 'MLCP v0.1 K=16 L=1'})
    bundle.save(a.output); print('saved', a.output, flush=True)
    sample = te[:a.check]; got = [bundle.diagnose(a.dataset / records[i]['file']) for i in sample]
    answered = [(g, y[i]) for g, i in zip(got, sample) if g['decision'] != 'abstain']
    report = {'checked_test_images': len(sample), 'abstained': len(sample) - len(answered),
              'accuracy_answered': float(np.mean([LABELS.index(g['decision']) == t for g, t in answered])) if answered else None,
              'tentative_accuracy': float(np.mean([LABELS.index(g['tentative_label']) == y[i] for g, i in zip(got, sample)]))}
    (a.output.parent / 'bundle-check.json').write_text(json.dumps(report, indent=2)); print(report)


if __name__ == '__main__':
    main()
