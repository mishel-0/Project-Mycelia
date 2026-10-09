"""Training-only small-data ablation for oriented cues and spatial topology.

No Testing image or label is loaded. This is a preliminary architecture gate,
not a held-out diagnostic result and not a substitute for a full refit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
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
from mycelia.omnia_image import load_omnia_rgb
from mycelia.visual_memory import (VisualConfig, VisualMycelium,
                                   prepare_gray_fixed, visual_cues)
from mycelia.visual_readout import spatial_pyramid_features
from conventional_omnia_baseline import hog_intensity
from medical_image_test import fit_centroids, metrics, predict
from omnia_tumor_research import LABELS, labels_for, stratified_validation, write_json


def load_clean_training(dataset_root, sdk_path):
    """Decode Training only and remove exact source/32px-cue duplicates."""
    sys.path.insert(0, str(Path(sdk_path).resolve()))
    with (Path(dataset_root)/'manifest.csv').open(newline='') as stream:
        manifest = [row for row in csv.DictReader(stream) if row['split'] == 'Training']
    records, seen_source, seen_cue = [], set(), set()
    removed = Counter()
    for row in manifest:
        if row['label'] not in LABELS:
            raise ValueError('invalid Training label')
        relative = Path(row['omnia']).relative_to('dataset').as_posix()
        image = load_omnia_rgb(Path(dataset_root)/row['omnia'])
        gray = prepare_gray_fixed(image, 32)
        cue = visual_cues(gray, detailed=True, multiscale=True)
        cue_hash = hashlib.sha256(np.ascontiguousarray(cue).tobytes()).hexdigest()
        source_hash = row['decoded_rgb_sha256']
        if source_hash in seen_source or cue_hash in seen_cue:
            removed['Training_duplicate'] += 1
            continue
        seen_source.add(source_hash); seen_cue.add(cue_hash)
        records.append({'file': relative, 'split': 'Training', 'label': row['label'],
                        'source_sha256': source_hash, 'cue_sha256': cue_hash})
    return records, np.arange(len(records), dtype=int), {
        'raw_training': len(manifest), 'training_unique': len(records),
        'removed': dict(removed), 'testing_files_decoded': 0,
    }


def balanced_sample(records, indices, per_class, rng):
    chosen = []
    for label in LABELS:
        pool = np.asarray([int(i) for i in indices if records[int(i)]['label'] == label])
        if len(pool) < per_class:
            raise ValueError(f'not enough {label} images for requested probe')
        chosen.extend(rng.choice(pool, per_class, replace=False))
    return np.asarray(chosen, dtype=int)


def cues_for(records, indices, dataset_root, size, mode):
    gray, cues = [], []
    for index in indices:
        image = load_omnia_rgb(Path(dataset_root)/'dataset'/records[int(index)]['file'])
        value = prepare_gray_fixed(image, size)
        gray.append(value)
        cues.append(visual_cues(value, detailed=True, multiscale=True, mode=mode))
    return np.asarray(gray), np.asarray(cues)


def train(cues, config, seed):
    model = VisualMycelium(config)
    started = time.perf_counter()
    for index in np.random.default_rng(seed).permutation(len(cues)):
        model.learn(cues[int(index)])
    model.freeze(); model.validate()
    return model, time.perf_counter()-started


def score_readouts(train_x, train_y, validation_x, validation_y):
    centroid = fit_centroids(train_x, train_y)
    centroid_score = metrics(validation_y, predict(centroid, validation_x))
    svm = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale', kernel='rbf',
                                               class_weight='balanced', cache_size=2048))
    started = time.perf_counter(); svm.fit(train_x, train_y)
    svm_fit = time.perf_counter()-started
    started = time.perf_counter(); svm_prediction = svm.predict(validation_x)
    svm_predict = time.perf_counter()-started
    return {'nearest_centroid': centroid_score,
            'fixed_rbf_svm': metrics(validation_y, svm_prediction),
            'svm_fit_seconds': svm_fit, 'svm_validation_seconds': svm_predict}


def run(dataset_root, sdk_path, output, train_per_class=75,
        validation_per_class=100, capacity=256, seed=20261008,
        only_size=0, only_mode=''):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records, clean_train, audit = load_clean_training(dataset_root, sdk_path)
    inner, validation_pool = stratified_validation(records, clean_train, seed=42)
    rng = np.random.default_rng(seed)
    training = balanced_sample(records, inner, train_per_class, rng)
    validation = balanced_sample(records, validation_pool, validation_per_class, rng)
    train_y, validation_y = labels_for(records, training), labels_for(records, validation)
    all_indices = np.concatenate((training, validation))
    results = []
    feature_sets = {}
    overall = time.perf_counter()
    configurations = ((32, 'edge_texture'), (32, 'oriented'),
                      (64, 'edge_texture'), (64, 'oriented'),
                      (64, 'axis_edges'))
    configurations = tuple((size, mode) for size, mode in configurations
                           if (not only_size or size == only_size) and
                              (not only_mode or mode == only_mode))
    if not configurations:
        raise ValueError('configuration filter selected nothing')
    for size, mode in configurations:
        gray, cues = cues_for(records, all_indices, dataset_root, size, mode)
        split = len(training); train_cues, validation_cues = cues[:split], cues[split:]
        config = VisualConfig(size=size, patch_size=8, max_colonies=capacity,
                              max_episodes=len(training), preprocessing='fixed',
                              multiscale_features=True, cue_mode=mode,
                              hierarchical_context=False, specialization=True,
                              error_gated=True)
        model, fit_seconds = train(train_cues, config, seed)
        levels = (1, 2, 4) if size == 32 else (1, 2, 4, 8)
        started = time.perf_counter()
        train_features = spatial_pyramid_features(model, train_cues, levels=levels)
        validation_features = spatial_pyramid_features(model, validation_cues, levels=levels)
        feature_seconds = time.perf_counter()-started
        readouts = score_readouts(train_features, train_y, validation_features, validation_y)
        feature_sets[(size, mode)] = (train_features, validation_features)
        reconstruction_mse = np.mean([
            np.mean((model.reconstruct(cue)[0]-cue)**2) for cue in validation_cues])
        row = {'size': size, 'cue_mode': mode, 'patch_size': 8,
               'patch_grid': size//8, 'levels': list(levels),
               'capacity': capacity, 'colonies': model.colony_count,
               'feature_dimension': train_features.shape[1],
               'unlabeled_fit_seconds': fit_seconds,
               'feature_seconds': feature_seconds,
               'validation_cue_reconstruction_mse': float(reconstruction_mse),
               'learning': {key: value for key, value in model.summary().items()
                            if key not in ('current_colony_support', 'selections')},
               'readouts': readouts}
        results.append(row)
        write_json(output/'progress.json', {'complete_configurations': len(results),
                   'total_configurations': len(configurations), 'latest': row})
        print(json.dumps({'size': size, 'mode': mode,
              'centroid_balanced_accuracy': readouts['nearest_centroid']['balanced_accuracy'],
              'svm_balanced_accuracy': readouts['fixed_rbf_svm']['balanced_accuracy'],
              'fit_seconds': fit_seconds}), flush=True)
    fusions = []
    fusion_specs = (
            ('32px_dual_receptor', ((32, 'edge_texture'), (32, 'oriented'))),
            ('64px_dual_receptor', ((64, 'edge_texture'), (64, 'oriented'))),
            ('64px_texture_axis_receptors', ((64, 'edge_texture'), (64, 'axis_edges'))),
            ('64px_all_receptors', ((64, 'edge_texture'), (64, 'oriented'),
                                    (64, 'axis_edges'))),
            ('edge_texture_multiresolution', ((32, 'edge_texture'), (64, 'edge_texture'))),
            ('dual_receptor_multiresolution', tuple(feature_sets)))
    for name, members in fusion_specs:
        if len(members) < 2 or any(key not in feature_sets for key in members):
            continue
        train_fused = np.concatenate([feature_sets[key][0] for key in members], axis=1)
        validation_fused = np.concatenate([feature_sets[key][1] for key in members], axis=1)
        readouts = score_readouts(train_fused, train_y, validation_fused, validation_y)
        fusions.append({'name': name, 'members': [list(key) for key in members],
                        'feature_dimension': train_fused.shape[1], 'readouts': readouts})
        print(json.dumps({'fusion': name,
              'centroid_balanced_accuracy': readouts['nearest_centroid']['balanced_accuracy'],
              'svm_balanced_accuracy': readouts['fixed_rbf_svm']['balanced_accuracy']}), flush=True)
    # Same chosen 32px images and label counts as an orientation-aware
    # conventional control. It never enters MYCELIA learning.
    gray32, _ = cues_for(records, all_indices, dataset_root, 32, 'oriented')
    conventional = score_readouts(hog_intensity(gray32[:len(training)]), train_y,
                                  hog_intensity(gray32[len(training):]), validation_y)
    result = {
        'schema': 'mycelia.training-only-orientation-ablation.v1',
        'scope': 'Training-only small-data architecture ablation; no Testing images or labels scored.',
        'seed': seed, 'capacity': capacity,
        'training_images': len(training), 'validation_images': len(validation),
        'training_per_class': dict(Counter(train_y)),
        'validation_per_class': dict(Counter(validation_y)),
        'labels_passed_to_visual_learning': False,
        'deduplication': audit, 'configurations': results,
        'label_free_feature_fusions': fusions,
        'matched_hog_intensity_control': conventional,
        'selection_rule_for_next_phase': 'Highest Training-only validation balanced accuracy; a full refit and fresh patient-grouped testing are still required.',
        'elapsed_seconds': time.perf_counter()-overall,
    }
    write_json(output/'summary.json', result)
    write_json(output/'progress.json', {'phase': 'complete'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--omnia-sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-per-class', type=int, default=75)
    parser.add_argument('--validation-per-class', type=int, default=100)
    parser.add_argument('--capacity', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--only-size', type=int, choices=(32, 64), default=0)
    parser.add_argument('--only-mode', choices=('edge_texture', 'oriented', 'axis_edges'),
                        default='')
    args = parser.parse_args()
    if min(args.train_per_class, args.validation_per_class, args.capacity) < 1:
        parser.error('counts and capacity must be positive')
    run(args.dataset_root, args.omnia_sdk, args.output, args.train_per_class,
        args.validation_per_class, args.capacity, args.seed,
        args.only_size, args.only_mode)


if __name__ == '__main__':
    main()
