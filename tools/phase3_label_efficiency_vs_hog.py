"""Phase 3 gate: does MYCELIA need fewer labels than HOG to reach a given accuracy?

Reuses the frozen 1,024-colony MYCELIA representation (already fit unlabeled on
all clean Training) and a fixed HOG+intensity descriptor. Both are read out
with the SAME nearest-centroid classifier at matched label-per-class counts,
so only the representation differs. Retrospective: Testing was scored before.
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
from mycelia.frozen_visual import FrozenVisualMatcher
from mycelia.visual_readout import colony_match_features
from medical_image_test import LABELS, fit_centroids, metrics, predict
from omnia_tumor_research import clean_image_splits, labels_for, load_omnia_dataset, read_test_labels
from conventional_omnia_baseline import hog_intensity

DEFAULT_COUNTS = (1, 2, 5, 10, 20, 40, 80, 160, 320)


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std())}


def run(dataset_root, sdk_path, compiled_path, output, repeats=20, seed=4200):
    started = time.perf_counter()
    records, cues, gray = load_omnia_dataset(dataset_root, sdk_path)
    train, test, audit = clean_image_splits(records)
    train_labels = labels_for(records, train)
    test_labels = read_test_labels(dataset_root, records, test)
    matcher = FrozenVisualMatcher.load(compiled_path)
    mycelia_train = colony_match_features(matcher, cues[train])
    mycelia_test = colony_match_features(matcher, cues[test])
    hog = hog_intensity(gray)
    hog_train, hog_test = hog[train], hog[test]
    class_indices = {label: np.flatnonzero(train_labels == label) for label in LABELS}
    maximum = min(map(len, class_indices.values()))
    counts = [n for n in DEFAULT_COUNTS if n <= maximum]
    rows = []
    for per_class in counts:
        mycelia_scores, hog_scores = [], []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 1009*per_class + repeat)
            chosen = np.concatenate([rng.choice(class_indices[label], per_class, replace=False)
                                     for label in LABELS])
            y = train_labels[chosen]
            m = fit_centroids(mycelia_train[chosen], y)
            h = fit_centroids(hog_train[chosen], y)
            mycelia_scores.append(metrics(test_labels, predict(m, mycelia_test))['balanced_accuracy'])
            hog_scores.append(metrics(test_labels, predict(h, hog_test))['balanced_accuracy'])
        row = {'labels_per_class': per_class, 'total_labeled_images': per_class*len(LABELS),
               'repeats': repeats,
               'mycelia_balanced_accuracy': summarize(mycelia_scores),
               'hog_balanced_accuracy': summarize(hog_scores),
               'mean_difference_mycelia_minus_hog': float(np.mean(mycelia_scores)-np.mean(hog_scores))}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('labels_per_class', 'mean_difference_mycelia_minus_hog')}
                         | {'mycelia': row['mycelia_balanced_accuracy']['mean'],
                            'hog': row['hog_balanced_accuracy']['mean']}), flush=True)
    result = {
        'schema': 'mycelia.phase3-label-efficiency-vs-hog.v1',
        'scope': __doc__.strip(),
        'gate': 'MYCELIA beats HOG+same-readout at <=40 labels/class',
        'deduplication': audit, 'repeats_per_point': repeats, 'seed_base': seed,
        'curve': rows,
        'gate_passed': any(r['labels_per_class'] <= 40 and r['mean_difference_mycelia_minus_hog'] > 0
                           for r in rows),
        'elapsed_seconds': time.perf_counter()-started,
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--omnia-sdk', type=Path, required=True)
    p.add_argument('--compiled-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--seed', type=int, default=4200)
    a = p.parse_args()
    run(a.dataset_root, a.omnia_sdk, a.compiled_model, a.output, a.repeats, a.seed)


if __name__ == '__main__':
    main()
