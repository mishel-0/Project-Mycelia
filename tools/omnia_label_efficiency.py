"""Retrospective label-count curves for a frozen unlabeled MYCELIA model.

The held-out set has already been evaluated, so this diagnoses label efficiency;
it is not a fresh confirmatory test and must not be used for model selection.
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
from omnia_tumor_research import (clean_image_splits, labels_for,
                                  load_omnia_dataset, read_test_labels)

DEFAULT_COUNTS = (1, 2, 5, 10, 20, 50, 100, 250, 500)


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'minimum': float(a.min()), 'maximum': float(a.max())}


def run(dataset_root, sdk_path, compiled_path, output, repeats=20, seed=4200):
    started = time.perf_counter()
    records, cues, _ = load_omnia_dataset(dataset_root, sdk_path)
    train, test, audit = clean_image_splits(records)
    train_labels = labels_for(records, train)
    test_labels = read_test_labels(dataset_root, records, test)
    matcher = FrozenVisualMatcher.load(compiled_path)
    train_features = colony_match_features(matcher, cues[train])
    test_features = colony_match_features(matcher, cues[test])
    raw_train = cues[train].reshape(len(train), -1)
    raw_test = cues[test].reshape(len(test), -1)
    class_indices = {label: np.flatnonzero(train_labels == label) for label in LABELS}
    maximum = min(map(len, class_indices.values()))
    counts = [n for n in DEFAULT_COUNTS if n <= maximum]
    rows = []
    for per_class in counts:
        learned_scores, raw_scores = [], []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 1009*per_class + repeat)
            chosen = np.concatenate([rng.choice(class_indices[label], per_class, replace=False)
                                     for label in LABELS])
            learned = fit_centroids(train_features[chosen], train_labels[chosen])
            raw = fit_centroids(raw_train[chosen], train_labels[chosen])
            learned_scores.append(metrics(test_labels, predict(learned, test_features))['balanced_accuracy'])
            raw_scores.append(metrics(test_labels, predict(raw, raw_test))['balanced_accuracy'])
        rows.append({'labels_per_class': per_class, 'total_labeled_images': per_class*len(LABELS),
                     'repeats': repeats, 'mycelia_balanced_accuracy': summarize(learned_scores),
                     'raw_cue_balanced_accuracy': summarize(raw_scores),
                     'mean_difference': float(np.mean(learned_scores)-np.mean(raw_scores))})
    full_learned = metrics(test_labels, predict(fit_centroids(train_features, train_labels), test_features))
    full_raw = metrics(test_labels, predict(fit_centroids(raw_train, train_labels), raw_test))
    result = {
        'schema': 'mycelia.retrospective-label-efficiency.v1',
        'scope': 'Exploratory replay on the previously evaluated image-level Testing set; no retraining of MYCELIA and no fresh confirmation.',
        'unlabeled_representation_images': len(train),
        'testing_images': len(test), 'repeats_per_point': repeats, 'seed_base': seed,
        'deduplication': audit, 'curve': rows,
        'all_training_labels': {'count': len(train),
            'mycelia_balanced_accuracy': full_learned['balanced_accuracy'],
            'raw_cue_balanced_accuracy': full_raw['balanced_accuracy']},
        'elapsed_seconds': time.perf_counter()-started,
        'interpretation': 'This measures labels needed by the external readout after MYCELIA already learned from all clean unlabeled Training images. It does not measure few-shot unlabeled representation learning.',
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--omnia-sdk', type=Path, required=True)
    parser.add_argument('--compiled-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1: parser.error('--repeats must be positive')
    print(json.dumps(run(args.dataset_root, args.omnia_sdk, args.compiled_model,
                         args.output, args.repeats), indent=2))


if __name__ == '__main__':
    main()
