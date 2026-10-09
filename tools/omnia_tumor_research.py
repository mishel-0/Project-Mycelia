"""Research-only, label-free MYCELIA representation with an external readout.

MYCELIA learns from OMNIA images without labels. A separate frozen NumPy
nearest-centroid readout maps the learned patch/colony match patterns to the
four labels in the supplied dataset. This is image-level research, not a
clinical diagnostic tool.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from mycelia.visual_memory import VisualConfig, VisualMycelium, visual_cues
from mycelia.visual_readout import colony_match_features, centroid_distance_scores
from unlabeled_visual_test import load_inputs
from medical_image_test import LABELS, fit_centroids, metrics, model_dict

CAPACITIES = (128, 256, 512, 1024)
SEED = 42


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def _hash_array(array):
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def load_omnia_dataset(dataset_root, sdk_path):
    dataset_root = Path(dataset_root)
    dataset_dir = dataset_root / 'dataset'
    manifest_path = dataset_root / 'manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'manifest not found: {manifest_path}')
    if sdk_path:
        sys.path.insert(0, str(Path(sdk_path).resolve()))
    manifest = list(csv.DictReader(manifest_path.open(newline='')))
    labels_by_path = {}
    for row in manifest:
        split = row.get('split')
        if split not in {'Training', 'Testing'}:
            raise ValueError(f'invalid split or label in manifest: {row}')
        if split == 'Training' and row.get('label') not in LABELS:
            raise ValueError(f'invalid Training label in manifest: {row}')
        key = Path(row['omnia']).relative_to('dataset').as_posix()
        if key in labels_by_path:
            raise ValueError(f'duplicate OMNIA path in manifest: {key}')
        labels_by_path[key] = (split, row['label'] if split == 'Training' else None)
    image_rows, gray32, _ = load_inputs(dataset_dir, preprocessing='fixed')
    records, cues = [], []
    for row, gray in zip(image_rows, gray32):
        key = Path(row['file']).as_posix()
        metadata = labels_by_path.get(key)
        if metadata is None:
            raise ValueError(f'OMNIA image has no manifest entry: {key}')
        split, training_label = metadata
        cue = visual_cues(gray, detailed=True, multiscale=True)
        records.append({
            'file': key,
            'split': split,
            # Testing labels stay out of the modeling records until the frozen
            # learner and readout are saved and the final evaluation begins.
            'label': training_label,
            'source_sha256': row['source_sha256'],
            'cue_sha256': _hash_array(cue),
        })
        cues.append(cue)
    if set(labels_by_path) != {r['file'] for r in records}:
        raise ValueError('manifest and OMNIA file set differ')
    return records, np.asarray(cues, dtype=np.float64), gray32


def clean_image_splits(records):
    """Remove duplicate inputs without using Testing labels for model selection."""
    training = [i for i, r in enumerate(records) if r['split'] == 'Training']
    testing = [i for i, r in enumerate(records) if r['split'] == 'Testing']
    conflicts = set()
    for field in ('source_sha256', 'cue_sha256'):
        labels = {}
        for i in training:
            labels.setdefault(records[i][field], set()).add(records[i]['label'])
        conflicts.update(key for key, value in labels.items() if len(value) > 1)
    train, train_source, train_cue = [], set(), set()
    removed = Counter()
    for i in training:
        r = records[i]
        hs, hc = r['source_sha256'], r['cue_sha256']
        if hs in conflicts or hc in conflicts:
            removed['Training_label_conflict'] += 1
        elif hs in train_source or hc in train_cue:
            removed['Training_duplicate'] += 1
        else:
            train.append(i)
        train_source.add(hs); train_cue.add(hc)
    test, test_source, test_cue = [], set(), set()
    for i in testing:
        r = records[i]
        hs, hc = r['source_sha256'], r['cue_sha256']
        if hs in train_source or hc in train_cue:
            removed['Testing_train_overlap'] += 1
        elif hs in test_source or hc in test_cue:
            removed['Testing_duplicate'] += 1
        else:
            test.append(i)
        test_source.add(hs); test_cue.add(hc)
    if not train or not test:
        raise ValueError('duplicate audit produced an empty Training or Testing split')
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int), {
        'removed': dict(removed),
        'training_label_conflicts': len(conflicts),
        'training_unique': len(train),
        'testing_unique': len(test),
        'test_labels_not_used_for_duplicate_filtering': True,
    }


def stratified_validation(records, training, seed=SEED, fraction=.2):
    rng = np.random.default_rng(seed)
    inner, validation = [], []
    for label in LABELS:
        indices = np.asarray([i for i in training if records[i]['label'] == label], dtype=int)
        if len(indices) < 2:
            raise ValueError(f'not enough Training images for {label}')
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        validation.extend(indices[:count])
        inner.extend(indices[count:])
    return np.asarray(inner, dtype=int), np.asarray(validation, dtype=int)


def _new_model(capacity):
    return VisualMycelium(VisualConfig(
        size=32, patch_size=8, max_colonies=capacity,
        preprocessing='fixed', multiscale_features=True,
        hierarchical_context=True, specialization=True, error_gated=True))


def fit_unlabeled(cues, indices, capacity, output, phase, seed=SEED):
    """Fit from cue arrays only: labels and records are deliberately absent."""
    model = _new_model(capacity)
    order = np.random.default_rng(seed).permutation(indices)
    started = time.monotonic()
    for step, index in enumerate(order, 1):
        model.learn(cues[int(index)])
        if (step % 250 == 0) or step == len(order):
            progress = {
                'phase': phase, 'capacity': capacity,
                'images_seen': model.images_seen, 'images_total': len(order),
                'colonies': len(model.colonies), 'updates': model.updates,
                'suppressed_updates': model.suppressed_updates,
                'novelty_overflow': model.novelty_overflow,
            }
            write_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    model.freeze()
    model.validate()
    evidence = model.summary()
    evidence.update({
        'images_seen': model.images_seen,
        'patches_seen': model.patches_seen,
        'epochs': 1,
        'labels_passed_to_learn': False,
        'fit_seconds': time.monotonic()-started,
    })
    if model.images_seen != len(indices):
        raise RuntimeError('MYCELIA did not complete every requested unlabeled exposure')
    return model, evidence


def labels_for(records, indices):
    return np.asarray([records[int(i)]['label'] for i in indices])


def read_test_labels(dataset_root, records, indices):
    """Read held-out labels only after model and readout artifacts are frozen."""
    with (Path(dataset_root)/'manifest.csv').open(newline='') as stream:
        rows = csv.DictReader(stream)
        labels = {Path(row['omnia']).relative_to('dataset').as_posix(): row['label']
                  for row in rows if row['split'] == 'Testing'}
    selected = [labels.get(records[int(i)]['file']) for i in indices]
    if any(label not in LABELS for label in selected):
        raise ValueError('a clean Testing image has no valid held-out label')
    return np.asarray(selected)


def _serial_metrics(actual, guesses):
    result = metrics(actual, guesses)
    result['confusion_labels'] = list(result['confusion_labels'])
    return result


def _readout_scores(readout, features):
    return centroid_distance_scores(readout, features)


def _predict_scores(readout, features):
    scores = _readout_scores(readout, features)
    guesses = np.asarray(readout['labels'])[np.argmin(scores, axis=1)]
    margins = np.sort(scores, axis=1)[:, 1] - np.sort(scores, axis=1)[:, 0]
    return guesses, scores, margins


def _save_predictions(path, records, indices, actual, guesses, scores, labels):
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file', 'actual', 'prediction', 'correct', 'nearest_centroid_distance',
                         'distance_margin', *[f'distance_{label}' for label in labels]])
        margins = np.sort(scores, axis=1)[:, 1] - np.sort(scores, axis=1)[:, 0]
        minima = scores.min(axis=1)
        for row, i in enumerate(indices):
            writer.writerow([records[int(i)]['file'], actual[row], guesses[row],
                             int(actual[row] == guesses[row]), float(minima[row]),
                             float(margins[row]), *[float(x) for x in scores[row]]])


def run(dataset_root, output, sdk_path=None, seed=SEED):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, cues, gray32 = load_omnia_dataset(dataset_root, sdk_path)
    clean_train, clean_test, audit = clean_image_splits(records)
    inner, validation = stratified_validation(records, clean_train, seed)
    write_json(output/'input-audit.json', {
        'images_loaded': len(records), 'raw_counts': dict(Counter(r['split'] for r in records)),
        'training_label_counts': dict(Counter(r['label'] for r in records if r['split'] == 'Training')),
        'deduplication': audit, 'training_inner': len(inner), 'training_validation': len(validation),
        'test_split_unseen_during_selection': len(clean_test),
    })
    print(f"Loaded {len(records)} OMNIA images; clean Training={len(clean_train)}, "
          f"Testing={len(clean_test)}; inner={len(inner)}, validation={len(validation)}", flush=True)

    candidate_results = []
    for capacity in CAPACITIES:
        model, evidence = fit_unlabeled(cues, inner, capacity, output,
                                        f'capacity-{capacity}', seed)
        train_features = colony_match_features(model, cues[inner])
        validation_features = colony_match_features(model, cues[validation])
        readout = fit_centroids(train_features, labels_for(records, inner))
        prediction, _, _ = _predict_scores(readout, validation_features)
        result = {
            'capacity': capacity,
            'colony_count': len(model.colonies),
            'validation': _serial_metrics(labels_for(records, validation), prediction),
            'learning': evidence,
            'feature_schema': 'four 2x2 spatial winner-colony histograms plus top-three mismatch values for each 4x4 patch grid',
            'feature_dimension': int(train_features.shape[1]),
        }
        candidate_results.append(result)
        write_json(output/'capacity-selection.json', {
            'seed': seed, 'inner_training_images': len(inner), 'validation_images': len(validation),
            'candidates': candidate_results,
            'selection_rule': 'Highest Training-only validation balanced accuracy; ties choose the smaller colony cap.',
        })
        print(f"capacity={capacity} validation balanced accuracy="
              f"{result['validation']['balanced_accuracy']:.4f}", flush=True)
        del model, train_features, validation_features, readout, prediction

    chosen = max(candidate_results,
                 key=lambda row: (row['validation']['balanced_accuracy'], -row['capacity']))
    capacity = int(chosen['capacity'])
    model, evidence = fit_unlabeled(cues, clean_train, capacity, output,
                                    'final-all-clean-training', seed)
    train_features = colony_match_features(model, cues[clean_train])
    readout = fit_centroids(train_features, labels_for(records, clean_train))
    model_path = output/'unsupervised-mycelia-memory.json'
    readout_path = output/'external-nearest-centroid-readout.json'
    model.save(model_path)
    readout_data = model_dict(readout)
    readout_data.update({
        'schema': 'mycelia.omnia-colony-centroid.v1',
        'feature_schema': 'mycelia-colony-match-features.v1',
        'feature_dimension': int(train_features.shape[1]),
        'labels': list(LABELS),
        'memory_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'scope': 'External Training-label readout over frozen label-free MYCELIA colony-match features; distance scores are not probabilities.',
    })
    write_json(readout_path, readout_data)
    # All controls are fixed using Training only before held-out scoring. The
    # three-channel control receives exactly the cues supplied to MYCELIA;
    # grayscale alone remains a separately reported, simpler comparison.
    raw_readout = fit_centroids(gray32[clean_train].reshape(len(clean_train), -1),
                               labels_for(records, clean_train))
    cue_readout = fit_centroids(cues[clean_train].reshape(len(clean_train), -1),
                               labels_for(records, clean_train))
    majority = Counter(labels_for(records, clean_train)).most_common(1)[0][0]
    permuted_labels = np.random.default_rng(seed).permutation(labels_for(records, clean_train))
    permuted_readout = fit_centroids(train_features, permuted_labels)
    write_json(output/'raw-cue-centroid-readout.json', model_dict(cue_readout))
    frozen_hash = hashlib.sha256(json.dumps(model.state_dict(), sort_keys=True,
                                            allow_nan=False).encode()).hexdigest()

    # Testing labels are first read here, after representation and readout freeze.
    test_features = colony_match_features(model, cues[clean_test])
    test_actual = read_test_labels(dataset_root, records, clean_test)
    test_prediction, test_scores, test_margins = _predict_scores(readout, test_features)
    if frozen_hash != hashlib.sha256(json.dumps(model.state_dict(), sort_keys=True,
                                                allow_nan=False).encode()).hexdigest():
        raise RuntimeError('frozen MYCELIA state changed during test inference')
    visual_metrics = _serial_metrics(test_actual, test_prediction)

    # Direct gray pixels and majority are controls evaluated on the same clean split.
    raw_prediction, raw_scores, _ = _predict_scores(raw_readout,
                                                      gray32[clean_test].reshape(len(clean_test), -1))
    raw_metrics = _serial_metrics(test_actual, raw_prediction)
    cue_prediction, cue_scores, _ = _predict_scores(
        cue_readout, cues[clean_test].reshape(len(clean_test), -1))
    cue_metrics = _serial_metrics(test_actual, cue_prediction)
    majority_metrics = _serial_metrics(test_actual, np.full(len(clean_test), majority))
    permuted_prediction, _, _ = _predict_scores(permuted_readout, test_features)
    permuted_metrics = _serial_metrics(test_actual, permuted_prediction)

    _save_predictions(output/'heldout-predictions.csv', records, clean_test,
                      test_actual, test_prediction, test_scores, LABELS)
    _save_predictions(output/'raw-pixel-baseline-predictions.csv', records, clean_test,
                      test_actual, raw_prediction, raw_scores, LABELS)
    _save_predictions(output/'raw-cue-baseline-predictions.csv', records, clean_test,
                      test_actual, cue_prediction, cue_scores, LABELS)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    restored = VisualMycelium.load(model_path)
    restored_features = colony_match_features(restored, cues[clean_test[:min(16, len(clean_test))]])
    roundtrip = np.array_equal(restored_features,
                               colony_match_features(model, cues[clean_test[:min(16, len(clean_test))]]))
    restored_prediction, _, _ = _predict_scores(readout, restored_features)
    roundtrip &= np.array_equal(restored_prediction, test_prediction[:len(restored_prediction)])
    single_check = True
    if len(clean_test):
        from mycelia.omnia_image import load_omnia_rgb
        test_path = Path(dataset_root)/'dataset'/records[int(clean_test[0])]['file']
        image = load_omnia_rgb(test_path)
        cue = model.encode_image(image)
        one_feature = colony_match_features(model, cue[None, ...])
        one_prediction, _, _ = _predict_scores(readout, one_feature)
        single_check = str(one_prediction[0]) == str(test_prediction[0])

    summary = {
        'schema': 'mycelia.omnia-unsupervised-readout.v1',
        'scope': 'Research-only four-class image-level prediction. Not a clinical diagnosis, patient-level test, or tumor localization result.',
        'dataset': str(Path(dataset_root).resolve()),
        'seed': seed,
        'raw_images': len(records),
        'clean_training_images': len(clean_train),
        'clean_testing_images': len(clean_test),
        'deduplication': audit,
        'capacity_selection': {'candidates': candidate_results, 'selected_capacity': capacity,
                               'rule': 'Highest Training-only validation balanced accuracy; ties choose the smaller colony cap.'},
        'final_learning': evidence,
        'representation': {
            'learned_images': model.images_seen,
            'labels_passed_to_visual_learning': False,
            'feature_schema': 'mycelia-colony-match-features.v1',
            'feature_dimension': int(train_features.shape[1]),
            'images_seen_by_final_readout': len(clean_train),
        },
        'heldout': {
            'scope': 'Frozen representation and readout; Testing labels accessed only after both were saved.',
            'mycelia_colony_readout': visual_metrics,
            'raw_32x32_gray_centroid_baseline': raw_metrics,
            'raw_32x32_three_channel_cue_centroid_baseline': cue_metrics,
            'training_majority_baseline': majority_metrics,
            'permuted_training_label_control': permuted_metrics,
            'distance_margin_summary': {
                'mean_nearest_centroid_squared_distance': float(np.mean(test_scores.min(axis=1))),
                'median_nearest_centroid_squared_distance': float(np.median(test_scores.min(axis=1))),
                'mean_distance_margin': float(np.mean(test_margins)),
                'interpretation': 'Scores and margins are standardized squared distances, not calibrated probabilities.',
            },
        },
        'artifacts': {
            'memory': model_path.name, 'memory_sha256': model_sha,
            'readout': readout_path.name,
        },
        'checks': {
            'all_clean_training_images_learned': model.images_seen == len(clean_train),
            'labels_not_passed_to_visual_learning': True,
            'heldout_frozen_state_unchanged': True,
            'checkpoint_roundtrip_features_and_predictions_identical': bool(roundtrip),
            'single_omnia_prediction_matches_batch': bool(single_check),
            'no_cnn_or_neural_network': True,
        },
        'elapsed_seconds': time.monotonic()-started,
    }
    summary['checks_passed'] = all(summary['checks'].values())
    write_json(output/'summary.json', summary)
    write_json(output/'progress.json', {'phase': 'complete', 'checks_passed': summary['checks_passed']})
    print(json.dumps({
        'selected_capacity': capacity,
        'mycelia_readout': visual_metrics,
        'raw_pixel_baseline': raw_metrics,
        'majority_baseline_accuracy': majority_metrics['accuracy'],
        'checks': summary['checks'],
        'elapsed_seconds': summary['elapsed_seconds'],
    }, indent=2), flush=True)
    if not summary['checks_passed']:
        raise RuntimeError('research training integrity check failed')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True,
                        help='Directory containing manifest.csv and dataset/{Training,Testing}/*.omnia')
    parser.add_argument('--omnia-sdk', type=Path, required=True,
                        help='Path to the omnia-sdk source checkout')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    run(args.dataset_root, args.output, args.omnia_sdk, args.seed)


if __name__ == '__main__':
    main()
