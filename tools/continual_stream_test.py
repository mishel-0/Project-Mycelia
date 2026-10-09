"""Phase 2: continual (streaming) learning on the OMNIA brain-MRI split.

Training images arrive in chunks. After each chunk every arm is scored on the
clean Testing images of the classes seen so far. Arms:
  mycelia_stale    MYCELIA learns the chunk unlabeled; a running class-mean
                   readout absorbs the chunk's features once (no replay).
  mycelia_refresh  same MYCELIA model; centroids recomputed on all seen data
                   with the current representation (measures drift cost).
  hog_running      fixed HOG descriptor + running class means (no replay).
  hog_svm_retrain  HOG + RBF-SVM retrained from scratch on all seen data.
  hog_svm_latest   HOG + RBF-SVM trained on the latest chunk only.
Retrospective: Testing was scored in earlier experiments.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))

import numpy as np
from mycelia.visual_readout import colony_match_features
from omnia_tumor_research import (LABELS, _new_model, clean_image_splits, labels_for,
                                  load_omnia_dataset, read_test_labels, write_json)
from conventional_omnia_baseline import hog_intensity
from matched_readout_comparison import svm


def padded_features(model, cues, capacity):
    """Colony-match features padded to a fixed width while the bank grows."""
    raw = colony_match_features(model, cues)
    k = model.colony_count
    out = np.zeros((len(cues), 4*capacity+48), dtype=np.float32)
    out[:, :4*capacity].reshape(len(cues), 4, capacity)[:, :, :k] = raw[:, :4*k].reshape(len(cues), 4, k)
    out[:, 4*capacity:] = raw[:, 4*k:]
    return out


class RunningCentroids:
    """Standardized nearest class mean from running sums; never revisits data."""

    def __init__(self, width):
        self.n = 0; self.s = np.zeros(width); self.ss = np.zeros(width)
        self.cn = {}; self.cs = {}

    def update(self, x, y):
        x = np.asarray(x, dtype=float)
        self.n += len(x); self.s += x.sum(0); self.ss += (x*x).sum(0)
        for label in set(y):
            rows = x[np.asarray(y) == label]
            self.cn[label] = self.cn.get(label, 0)+len(rows)
            self.cs[label] = self.cs.get(label, 0)+rows.sum(0)

    def predict(self, x):
        mean = self.s/self.n
        scale = np.sqrt(np.maximum(self.ss/self.n-mean*mean, 0)); scale[scale < 1e-12] = 1
        labels = sorted(self.cn)
        centroids = np.stack([(self.cs[c]/self.cn[c]-mean)/scale for c in labels])
        z = (np.asarray(x, dtype=float)-mean)/scale
        d = ((z[:, None, :]-centroids[None])**2).sum(2)
        return np.asarray(labels)[np.argmin(d, axis=1)]


def make_chunks(records, train, protocol, n_chunks, rng):
    if protocol == 'iid':
        return np.array_split(rng.permutation(train), n_chunks)
    per_class = max(1, n_chunks//len(LABELS))
    chunks = []
    for label in LABELS:
        pool = rng.permutation([i for i in train if records[int(i)]['label'] == label])
        chunks.extend(np.array_split(pool, per_class))
    return chunks


def accuracy(actual, predicted):
    return float(np.mean(np.asarray(actual) == np.asarray(predicted))) if len(actual) else None


def run_protocol(protocol, records, cues, hog, train, test, test_labels, capacity, n_chunks, seed):
    rng = np.random.default_rng(seed)
    chunks = make_chunks(records, train, protocol, n_chunks, rng)
    model = _new_model(capacity)
    width = 4*capacity+48
    stale, hog_run = RunningCentroids(width), RunningCentroids(hog.shape[1])
    seen, seen_classes, rows = [], set(), []
    cost = {k: 0.0 for k in ('mycelia_learn', 'mycelia_refresh', 'hog_svm_retrain')}
    for step, chunk in enumerate(chunks, 1):
        chunk = np.asarray(chunk, dtype=int)
        y = labels_for(records, chunk)
        t = time.perf_counter()
        for i in chunk:
            model.learn(cues[int(i)])
        stale.update(padded_features(model, cues[chunk], capacity), y)
        cost['mycelia_learn'] += time.perf_counter()-t
        hog_run.update(hog[chunk], y)
        seen.extend(chunk.tolist()); seen_classes.update(y)
        seen_arr = np.asarray(seen)
        mask = np.isin(test_labels, sorted(seen_classes))
        t_idx, t_y = test[mask], test_labels[mask]
        test_x = padded_features(model, cues[t_idx], capacity)
        t = time.perf_counter()
        refresh = RunningCentroids(width)
        refresh.update(padded_features(model, cues[seen_arr], capacity), labels_for(records, seen_arr))
        cost['mycelia_refresh'] += time.perf_counter()-t
        row = {'chunk': step, 'images_seen': len(seen), 'classes_seen': sorted(seen_classes),
               'test_images': int(mask.sum()), 'colonies': model.colony_count,
               'mycelia_stale': accuracy(t_y, stale.predict(test_x)),
               'mycelia_refresh': accuracy(t_y, refresh.predict(test_x)),
               'hog_running': accuracy(t_y, hog_run.predict(hog[t_idx]))}
        if len(seen_classes) > 1:
            t = time.perf_counter()
            retrain = svm(10.0).fit(hog[seen_arr], labels_for(records, seen_arr))
            cost['hog_svm_retrain'] += time.perf_counter()-t
            row['hog_svm_retrain'] = accuracy(t_y, retrain.predict(hog[t_idx]))
        else:
            row['hog_svm_retrain'] = None
        if len(set(y)) > 1:
            latest = svm(10.0).fit(hog[chunk], y)
            row['hog_svm_latest'] = accuracy(t_y, latest.predict(hog[t_idx]))
        else:
            row['hog_svm_latest'] = None
        row['cumulative_seconds'] = dict(cost)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('chunk', 'images_seen', 'mycelia_stale',
              'mycelia_refresh', 'hog_running', 'hog_svm_retrain', 'hog_svm_latest')}), flush=True)
    # Forgetting: accuracy on each class's own test images right after its
    # first chunk vs. at the end of the stream (class-incremental meaningful).
    return {'protocol': protocol, 'chunks': len(chunks), 'per_chunk': rows,
            'final_mycelia_learning': {k: v for k, v in model.summary().items()
                                       if k not in ('current_colony_support', 'selections')},
            'cumulative_seconds': cost}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--omnia-sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capacity', type=int, default=256)
    p.add_argument('--chunks', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--protocols', default='iid,class_incremental')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records, cues, gray = load_omnia_dataset(a.dataset_root, a.omnia_sdk)
    train, test, audit = clean_image_splits(records)
    test_labels = read_test_labels(a.dataset_root, records, test)
    hog = hog_intensity(gray)
    results = {'schema': 'mycelia.continual-stream.v1', 'scope': __doc__.strip(),
               'capacity': a.capacity, 'seed': a.seed, 'deduplication': audit,
               'labels_passed_to_visual_learning': False, 'protocols': {}}
    for protocol in a.protocols.split(','):
        results['protocols'][protocol] = run_protocol(protocol, records, cues, hog, train, test,
                                                      test_labels, a.capacity, a.chunks, a.seed)
        write_json(a.output/'summary.json', results)


if __name__ == '__main__':
    main()
