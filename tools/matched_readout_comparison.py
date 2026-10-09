"""Separate representation quality from readout choice on the OMNIA split.

Applies the identical Training-only RBF-SVM protocol used by the HOG baseline
to (a) frozen MYCELIA colony features, (b) HOG+intensity, and (c) both
concatenated. Retrospective: the Testing set was evaluated before. The frozen
MYCELIA memory was fit unlabeled on all clean Training images, so its features
for the inner validation partition were learned without labels but not held
out from representation learning; C selection uses labels from inner only.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from medical_image_test import metrics
from mycelia.frozen_visual import FrozenVisualMatcher
from mycelia.visual_readout import colony_match_features
from omnia_tumor_research import (clean_image_splits, labels_for, load_omnia_dataset,
                                  read_test_labels, stratified_validation)
from conventional_omnia_baseline import hog_intensity


def svm(c):
    return make_pipeline(StandardScaler(), SVC(C=c, gamma='scale', kernel='rbf',
                                               class_weight='balanced', cache_size=2048))


def evaluate(name, x, records, inner, validation, train, test, actual):
    candidates = []
    for c in (1.0, 10.0, 100.0):
        model = svm(c).fit(x[inner], labels_for(records, inner))
        score = metrics(labels_for(records, validation), model.predict(x[validation]))
        candidates.append({'C': c, 'validation_balanced_accuracy': score['balanced_accuracy']})
    chosen = max(candidates, key=lambda r: (r['validation_balanced_accuracy'], -r['C']))
    started = time.perf_counter()
    final = svm(chosen['C']).fit(x[train], labels_for(records, train))
    fit_seconds = time.perf_counter()-started
    prediction = final.predict(x[test])
    result = metrics(actual, prediction)
    print(f"{name}: dim={x.shape[1]} C={chosen['C']} test_acc={result['accuracy']:.4f} "
          f"bal={result['balanced_accuracy']:.4f}", flush=True)
    return {'feature_dimension': int(x.shape[1]), 'candidates': candidates,
            'selected_C': chosen['C'], 'final_fit_seconds': fit_seconds,
            'heldout_metrics': result}, prediction


def mcnemar(actual, a, b):
    a_ok, b_ok = a == actual, b == actual
    only_a, only_b = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    stat = (abs(only_a-only_b)-1)**2/n if n else 0.0
    return {'only_first_correct': only_a, 'only_second_correct': only_b,
            'chi2_continuity_corrected': stat}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--omnia-sdk', type=Path, required=True)
    p.add_argument('--compiled-model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    records, cues, gray = load_omnia_dataset(a.dataset_root, a.omnia_sdk)
    train, test, audit = clean_image_splits(records)
    inner, validation = stratified_validation(records, train, a.seed)
    matcher = FrozenVisualMatcher.load(a.compiled_model)
    mycelia_x = colony_match_features(matcher, cues)
    hog_x = hog_intensity(gray)
    actual = read_test_labels(a.dataset_root, records, test)
    out, preds = {}, {}
    for name, x in (('mycelia_svm', mycelia_x), ('hog_svm', hog_x),
                    ('mycelia_plus_hog_svm', np.hstack([mycelia_x, hog_x]))):
        out[name], preds[name] = evaluate(name, x, records, inner, validation, train, test, actual)
    result = {
        'schema': 'mycelia.matched-readout-comparison.v1',
        'scope': __doc__.strip(),
        'counts': {'inner': len(inner), 'validation': len(validation),
                   'train': len(train), 'test': len(test)},
        'deduplication': audit,
        'models': out,
        'paired': {
            'hog_vs_mycelia': mcnemar(actual, preds['hog_svm'], preds['mycelia_svm']),
            'combined_vs_hog': mcnemar(actual, preds['mycelia_plus_hog_svm'], preds['hog_svm']),
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['paired'], indent=2))


if __name__ == '__main__':
    main()
