"""Conventional non-neural HOG/intensity SVM baseline on the OMNIA split.

This is a retrospective comparator because the Testing set has already been
evaluated. Selection uses only the stratified Training validation partition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from medical_image_test import metrics
from omnia_tumor_research import (clean_image_splits, labels_for,
                                  load_omnia_dataset, read_test_labels,
                                  stratified_validation)


def hog_intensity(gray):
    """Fixed 580-value descriptor: 2x2-cell HOG blocks + 16px intensity."""
    gray = np.asarray(gray, dtype=np.float32)
    gy, gx = np.gradient(gray, axis=(1, 2))
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.minimum((angle*(9/np.pi)).astype(int), 8)
    cells = np.zeros((len(gray), 4, 4, 9), dtype=np.float32)
    for cy in range(4):
        for cx in range(4):
            ys, xs = slice(cy*8, (cy+1)*8), slice(cx*8, (cx+1)*8)
            for b in range(9):
                cells[:, cy, cx, b] = np.sum(
                    magnitude[:, ys, xs]*(bins[:, ys, xs] == b), axis=(1, 2))
    blocks = []
    for cy in range(3):
        for cx in range(3):
            block = cells[:, cy:cy+2, cx:cx+2].reshape(len(gray), -1)
            block /= np.sqrt(np.sum(block*block, axis=1, keepdims=True)+1e-8)
            blocks.append(block)
    intensity = gray.reshape(len(gray), 16, 2, 16, 2).mean(axis=(2, 4)).reshape(len(gray), -1)
    return np.concatenate([*blocks, intensity], axis=1)


def run(dataset_root, sdk_path, output, seed=42):
    overall = time.perf_counter()
    records, _, gray = load_omnia_dataset(dataset_root, sdk_path)
    train, test, audit = clean_image_splits(records)
    inner, validation = stratified_validation(records, train, seed)
    started = time.perf_counter()
    x = hog_intensity(gray)
    feature_seconds = time.perf_counter()-started
    candidates = []
    for c in (1.0, 10.0, 100.0):
        model = make_pipeline(StandardScaler(), SVC(C=c, gamma='scale', kernel='rbf',
                                                     class_weight='balanced', cache_size=2048))
        started = time.perf_counter()
        model.fit(x[inner], labels_for(records, inner))
        fit_seconds = time.perf_counter()-started
        score = metrics(labels_for(records, validation), model.predict(x[validation]))
        candidates.append({'C': c, 'fit_seconds': fit_seconds,
                           'validation_accuracy': score['accuracy'],
                           'validation_balanced_accuracy': score['balanced_accuracy']})
    chosen = max(candidates, key=lambda row: (row['validation_balanced_accuracy'], -row['C']))
    final = make_pipeline(StandardScaler(), SVC(C=chosen['C'], gamma='scale', kernel='rbf',
                                                 class_weight='balanced', cache_size=2048))
    started = time.perf_counter(); final.fit(x[train], labels_for(records, train))
    final_fit_seconds = time.perf_counter()-started
    actual = read_test_labels(dataset_root, records, test)
    started = time.perf_counter(); prediction = final.predict(x[test])
    prediction_seconds = time.perf_counter()-started
    score = metrics(actual, prediction)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        'schema': 'mycelia.conventional-comparator.v1',
        'scope': 'Retrospective non-neural comparator; Training-only selection, previously evaluated image-level Testing set.',
        'descriptor': 'Fixed 580-value HOG plus downsampled grayscale intensity',
        'classifier': 'standardized RBF support-vector classifier',
        'neural_network': False, 'seed': seed, 'deduplication': audit,
        'counts': {'inner_training': len(inner), 'validation': len(validation),
                   'all_training': len(train), 'testing': len(test)},
        'feature_extraction_seconds_all_images': feature_seconds,
        'candidates': candidates, 'selected_C': chosen['C'],
        'final_fit_seconds': final_fit_seconds,
        'testing_prediction_seconds': prediction_seconds,
        'heldout_metrics': score,
        'process_peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak*1024),
        'elapsed_seconds': time.perf_counter()-overall,
        'limits': 'Does not prove clinical performance or compare energy use. Same image-level provenance limits as MYCELIA.',
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--omnia-sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.omnia_sdk, args.output, args.seed), indent=2))


if __name__ == '__main__':
    main()
