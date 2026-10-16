"""Real drone imagery benchmark (AeroScapes): few-label patch recognition + adaptation.

Task: recognise what a small patch of a drone-camera image shows (road,
vegetation, building, ...), using the dataset's per-pixel masks only to label
patches. Compared at several label budgets, 10 seeds, with identical labels:

  mycelia     : unsupervised receptor field on unlabelled patches + teach-only trace memory
  mobilenet   : ImageNet-pretrained MobileNetV3-small features + logistic regression (linear probe)
  cnn_scratch : small CNN with augmentation trained from scratch on the same labelled patches
  raw_svm     : RBF-SVM on raw gray pixels

MYCELIA and the scratch CNN see 16x16 grayscale patches; MobileNet gets colour and an upsampled
patch (it is the strongest conventional baseline). Splits are by drone sequence (no frames of one
flight on both sides). Stage 2 shifts the test domain (synthetic low-light + noise) and streams
unlabelled tiles with a few labels, comparing frozen / memory-only / receptor-adapting MYCELIA with a
refit linear probe. CPU only. Research benchmark, not flight software.
"""
from __future__ import annotations
import argparse, json, re, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mycelia.receptor_memory import ReceptorConfig, ReceptorField, TraceMemory, _norm
from swarm_vision_pod import TunedCNN

PATCH, W, H = 16, 256, 144


def sequence_of(stem):
    m = re.match(r'^(\d+)_', stem)
    return m.group(1) if m else None


def load_patches(root, max_images, min_purity, rng):
    jpg = sorted((root / 'JPEGImages').glob('*.jpg')); seg = root / 'SegmentationClass'
    print('images found:', len(jpg), 'e.g.', [p.stem for p in jpg[:4]], flush=True)
    if max_images and len(jpg) > max_images: jpg = [jpg[i] for i in sorted(rng.choice(len(jpg), max_images, replace=False))]
    gray, rgb, lab, grp = [], [], [], []
    for i, p in enumerate(jpg):
        mp = seg / (p.stem + '.png')
        if not mp.exists(): continue
        img = Image.open(p).convert('RGB').resize((W, H), Image.Resampling.BILINEAR)
        mask = np.asarray(Image.open(mp).resize((W, H), Image.Resampling.NEAREST))
        if mask.ndim == 3: mask = mask[..., 0]
        a = np.asarray(img, dtype=np.float32) / 255; g = a.mean(2)
        seq = sequence_of(p.stem) or f'block{i // 30}'
        for r in range(0, H - PATCH + 1, PATCH):
            for c in range(0, W - PATCH + 1, PATCH):
                m = mask[r:r + PATCH, c:c + PATCH].ravel(); top, n = Counter(m.tolist()).most_common(1)[0]
                if n / m.size >= min_purity:
                    gray.append(g[r:r + PATCH, c:c + PATCH]); rgb.append(a[r:r + PATCH, c:c + PATCH]); lab.append(top); grp.append(seq)
    return np.stack(gray).astype(np.float32), np.stack(rgb).astype(np.float32), np.array(lab), np.array(grp)


def mobilenet_features(rgb, batch=512):
    import torch, torchvision
    import os; torch.set_num_threads(max(1, os.cpu_count() or 1))
    net = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT).eval()
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb), batch):
            x = torch.tensor(rgb[i:i + batch]).permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, size=96, mode='bilinear', align_corners=False)
            f = net.avgpool(net.features((x - mean) / std)).flatten(1); out.append(f.numpy())
    return np.concatenate(out)


class ScratchCNN(TunedCNN):
    name = 'cnn_scratch'
    def __init__(self, k): super().__init__(.1); self.k = k
    def _net(self):
        import torch.nn as nn
        return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(.2), nn.Linear(32 * 16, 64), nn.ReLU(), nn.Linear(64, self.k))


def pick(y, per, rng, classes):
    return np.concatenate([rng.choice(np.where(y == c)[0], min(per, (y == c).sum()), replace=False) for c in classes])


def stage1(a, G, F_myc, F_mob, y, grp, classes, split):
    tr, te = split; k = len(classes); remap = {c: i for i, c in enumerate(classes)}; yy = np.array([remap.get(v, -1) for v in y])
    rows = []
    te_idx = np.concatenate([np.random.default_rng(0).choice(te[yy[te] == i], min(a.test_per_class, (yy[te] == i).sum()), replace=False) for i in range(k)])
    for per in a.labels:
        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 * per + seed); idx = pick(yy[tr], per, rng, range(k)); lab = tr[idx]
            res = {}
            t0 = time.perf_counter(); mem = TraceMemory(k, gamma=4., lam=.01).fit(F_myc[lab], yy[lab]); res['mycelia'] = (mem.predict(F_myc[te_idx]), time.perf_counter() - t0)
            t0 = time.perf_counter(); lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.)).fit(F_mob[lab], yy[lab]); res['mobilenet'] = (lr.predict(F_mob[te_idx]), time.perf_counter() - t0)
            t0 = time.perf_counter(); cnn = ScratchCNN(k).fit(G[lab], yy[lab], seed); res['cnn_scratch'] = (cnn.predict(G[te_idx]), time.perf_counter() - t0)
            t0 = time.perf_counter(); sv = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale')).fit(G[lab].reshape(len(lab), -1), yy[lab]); res['raw_svm'] = (sv.predict(G[te_idx].reshape(len(te_idx), -1)), time.perf_counter() - t0)
            for m, (p, secs) in res.items():
                t = yy[te_idx]; acc = float((p == t).mean()); bal = float(np.mean([(p[t == c] == c).mean() for c in range(k)]))
                rows.append({'labels_per_class': per, 'seed': seed, 'method': m, 'accuracy': acc, 'balanced_accuracy': bal, 'fit_secs': secs})
        print('stage1 labels/class', per, {m: round(float(np.mean([r['balanced_accuracy'] for r in rows if r['labels_per_class'] == per and r['method'] == m])), 3) for m in res}, flush=True)
    return rows


def darken(g, rng):
    return np.clip(g ** 1.8 * .8 + .06 * rng.standard_normal(g.shape), 0, 1).astype(np.float32)


def stage2(a, field, G, rgb, y, classes, split, F_mob_fn):
    """Domain shift: test world is dark and noisy. Stream unlabelled tiles + few labels."""
    import copy
    tr, te = split; k = len(classes); remap = {c: i for i, c in enumerate(classes)}; yy = np.array([remap.get(v, -1) for v in y]); rows = []
    for seed in range(a.shift_seeds):
        rng = np.random.default_rng(777 + seed); te_idx = np.concatenate([rng.choice(te[yy[te] == i], min(a.test_per_class, (yy[te] == i).sum()), replace=False) for i in range(k)])
        pool = np.setdiff1d(te, te_idx); lab0 = tr[pick(yy[tr], a.labels_A, rng, range(k))]
        dark_g = {int(i): darken(G[i], rng) for i in np.concatenate([te_idx, pool])}
        dark_rgb = np.stack([np.clip(rgb[i] ** 1.8 * .8, 0, 1) for i in te_idx])
        Xte_g = np.stack([dark_g[int(i)] for i in te_idx]); Fte_mob = F_mob_fn(dark_rgb); yte = yy[te_idx]
        bufs = {n: ([G[i] for i in lab0], [yy[i] for i in lab0]) for n in ('mycelia_memory', 'mycelia_adapt')}
        fields = {n: copy.deepcopy(field) for n in ('frozen_mycelia', 'mycelia_memory', 'mycelia_adapt')}
        mem = {n: TraceMemory(k, gamma=4., lam=.01).fit(fields[n].encode_many(list(bufs.get(n, bufs['mycelia_memory'])[0])), np.array(bufs.get(n, bufs['mycelia_memory'])[1])) for n in fields}
        lab_rgb0 = np.stack([rgb[i] for i in lab0]); probe_X = [F_mob_fn(lab_rgb0)]; probe_y = [yy[lab0]]
        def score(t):
            r = {'t': t}
            for n in fields: r[n] = float((mem[n].predict(fields[n].encode_many(list(Xte_g))) == yte).mean())
            pr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(np.concatenate(probe_X), np.concatenate(probe_y)); r['mobilenet_refit'] = float((pr.predict(Fte_mob) == yte).mean())
            return r
        curve = [score(0)]
        for t in range(1, a.batches + 1):
            b = rng.choice(pool, a.stream_batch, replace=False); xb = [dark_g[int(i)] for i in b]
            li = pick(yy[b], a.labels_per_batch, rng, range(k)) if all((yy[b] == c).any() for c in range(k)) else np.arange(0)
            for n in ('mycelia_memory', 'mycelia_adapt'):
                bx, by = bufs[n]
                for j in li: bx.append(xb[j]); by.append(int(yy[b][j]))
                # keep a class-balanced window of the most recent 60 per class
                keep = []
                for c in range(k):
                    ids = [q for q, v in enumerate(by) if v == c][-60:]; keep += ids
                keep.sort(); bufs[n] = ([bx[q] for q in keep], [by[q] for q in keep]); bx, by = bufs[n]
                if n == 'mycelia_adapt':
                    f = fields[n]; c = f.config
                    for img in xb:
                        p = f._patches(img).reshape(-1, c.patch * c.patch); p = p[rng.choice(len(p), min(15, len(p)), replace=False)]
                        z = (_norm(p) - f.mu) @ f.whiten; win = np.argmax(z @ f.receptors.T, 1)
                        for zi, kk in zip(z, win):
                            f.receptors[kk] += .03 * zi / (np.linalg.norm(zi) + 1e-9); f.receptors[kk] /= np.linalg.norm(f.receptors[kk]) + 1e-9
                mem[n] = TraceMemory(k, gamma=4., lam=.01).fit(fields[n].encode_many(list(bx)), np.array(by))
            if len(li):
                probe_X.append(F_mob_fn(np.clip(np.stack([rgb[b[j]] for j in li]) ** 1.8 * .8, 0, 1))); probe_y.append(yy[b][li])
            curve.append(score(t))
        rows.append(curve); print('stage2 seed', seed, {n: (round(curve[0][n], 3), round(curve[-1][n], 3)) for n in curve[0] if n != 't'}, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True); ap.add_argument('--output', type=Path, default=Path('runs/aeroscapes'))
    ap.add_argument('--max-images', type=int, default=1200); ap.add_argument('--min-purity', type=float, default=.9)
    ap.add_argument('--labels', type=int, nargs='+', default=[5, 10, 20, 40, 150]); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--min-class-patches', type=int, default=400); ap.add_argument('--test-per-class', type=int, default=300)
    ap.add_argument('--unlabeled', type=int, default=2000); ap.add_argument('--receptors', type=int, default=100)
    ap.add_argument('--shift-seeds', type=int, default=5); ap.add_argument('--batches', type=int, default=10)
    ap.add_argument('--stream-batch', type=int, default=90); ap.add_argument('--labels-per-batch', type=int, default=2); ap.add_argument('--labels-A', dest='labels_A', type=int, default=20)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(0)
    G, RGB, y, grp = load_patches(a.root, a.max_images, a.min_purity, rng)
    seqs = sorted(set(grp)); print('patches', len(y), 'sequences', len(seqs), 'class counts', dict(Counter(y.tolist())), f'{time.time()-t0:.0f}s', flush=True)
    if len(seqs) < 4: raise SystemExit('too few sequence groups for a grouped split')
    order = rng.permutation(len(seqs)); test_seqs = {seqs[i] for i in order[:max(1, len(seqs) // 4)]}
    is_te = np.array([g in test_seqs for g in grp]); tr, te = np.where(~is_te)[0], np.where(is_te)[0]
    cnt_tr, cnt_te = Counter(y[tr].tolist()), Counter(y[te].tolist())
    classes = sorted(c for c in cnt_tr if cnt_tr[c] >= a.min_class_patches and cnt_te.get(c, 0) >= a.test_per_class // 2)
    print('train/test patches', len(tr), len(te), 'classes used', classes, f'{time.time()-t0:.0f}s', flush=True)
    keep = np.isin(y, classes); tr, te = tr[keep[tr]], te[keep[te]]
    unl = tr[rng.choice(len(tr), min(a.unlabeled, len(tr)), replace=False)]
    field = ReceptorField(ReceptorConfig(receptors=a.receptors, patch=6, grid=2, patches_per_image=12, iterations=10)).fit([G[i] for i in unl])
    F_myc = field.encode_many(list(G)); F_mob = mobilenet_features(RGB); print('features done', f'{time.time()-t0:.0f}s', flush=True)
    rows1 = stage1(a, G, F_myc, F_mob, y, grp, classes, (tr, te))
    rows2 = stage2(a, field, G, RGB, y, classes, (tr, te), mobilenet_features)
    summ = {}
    for per in a.labels:
        d = {}
        for m in ('mycelia', 'mobilenet', 'cnn_scratch', 'raw_svm'):
            v = [r['balanced_accuracy'] for r in sorted((r for r in rows1 if r['labels_per_class'] == per and r['method'] == m), key=lambda r: r['seed'])]; d[m] = float(np.mean(v))
        diff = np.array([r['balanced_accuracy'] for r in sorted((r for r in rows1 if r['labels_per_class'] == per and r['method'] == 'mycelia'), key=lambda r: r['seed'])]) - \
               np.array([r['balanced_accuracy'] for r in sorted((r for r in rows1 if r['labels_per_class'] == per and r['method'] == 'mobilenet'), key=lambda r: r['seed'])])
        b = [np.random.default_rng(i).choice(diff, len(diff)).mean() for i in range(2000)]
        summ[per] = {'balanced_accuracy': d, 'mycelia_minus_mobilenet': float(diff.mean()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    names = [n for n in rows2[0][0] if n != 't']
    stream = {n: [float(np.mean([c[t][n] for c in rows2])) for t in range(a.batches + 1)] for n in names}
    out = {'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, 'classes': [int(c) for c in classes], 'class_counts_train': {int(k): int(v) for k, v in cnt_tr.items()},
           'stage1_summary': summ, 'stage1_fit_secs': {m: float(np.mean([r['fit_secs'] for r in rows1 if r['method'] == m])) for m in ('mycelia', 'mobilenet', 'cnn_scratch', 'raw_svm')},
           'stage2_stream_accuracy': stream, 'seconds': time.time() - t0}
    (a.output / 'results.json').write_text(json.dumps(out | {'rows': rows1, 'stage2_curves': rows2}, indent=2))
    print('RESULT_SUMMARY_BEGIN'); print(json.dumps(out)); print('RESULT_SUMMARY_END')


if __name__ == '__main__':
    main()
