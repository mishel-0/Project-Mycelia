#!/usr/bin/env python3
"""Patient-held-out full 3D patch reconstruction with MYCELIA colonies.

All 64 DCT modes in each 4x4x4 MRI patch are prediction targets. Colony
assignment sees those complete visual vectors, while local predictive maps
use compact DCT8 context cues to keep the topology and covariance models
tractable. No diagnostic labels enter learning or calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import rhuh_mri_pilot as pilot
from mycelia.volumetric_memory import (
    GroupedPredictiveMemory, VolumetricMemoryConfig, VolumetricMycelium,
)


def context_channels(modality_count):
    """Indices of low-frequency cues retained from every MRI sequence."""
    mode_index = {mode: index for index, mode in enumerate(pilot._DCT64_FREQUENCIES)}
    return tuple(modality*64+mode_index[mode]
                 for modality in range(modality_count)
                 for mode in pilot._DCT8_FREQUENCIES)


def full_dct_projection_mse(normalized_feature_mse, patch=4):
    """Convert errors for all orthonormal DCT modes to source-voxel MSE."""
    positions = np.arange(patch, dtype=np.float64)
    frequencies = np.arange(patch, dtype=np.float64)[:, None]
    basis = np.sqrt(2.0/patch)*np.cos(
        np.pi*(2*positions+1)*frequencies/(2*patch))
    basis[0] = 1.0/np.sqrt(patch)
    total = 0.0
    for (z, y, x), mse in zip(pilot._DCT64_FREQUENCIES, normalized_feature_mse):
        kernel = np.einsum('i,j,k->ijk', basis[z], basis[y], basis[x])
        bound = float(np.abs(kernel).sum())
        total += float(mse)*bound*bound
    return total/(patch**3)


def load_dataset(dataset, modalities, factor, patch):
    visits_by_patient = pilot.patient_visits(dataset, modalities)
    loaded = []
    for patient in sorted(visits_by_patient):
        for visit, files in visits_by_patient[patient]:
            X, visual, labels, coords, shape = pilot.load_visit(
                files, factor, patch, modalities, 'dct64')
            loaded.append((patient, visit, X, labels, coords, shape, visual))
    return visits_by_patient, loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--clusters', type=int, default=16)
    parser.add_argument('--max-patients', type=int, default=0)
    parser.add_argument('--modalities', default='auto')
    args = parser.parse_args()
    if args.patch != 4:
        parser.error('full DCT64 currently requires --patch 4')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.modalities == 'auto':
        modalities = pilot.MODALITIES
        index = args.dataset/'index.csv'
        if index.is_file():
            import csv
            with index.open(newline='') as stream:
                found = {pilot.MIRROR_MODALITIES.get(row.get('sequence', ''))
                         for row in csv.DictReader(stream)
                         if row.get('kind') == 'images'}
            modalities = tuple(item for item in pilot.MODALITIES if item in found)
    else:
        modalities = tuple(part.strip().lower() for part in args.modalities.split(',')
                           if part.strip())
        if any(item not in pilot.MODALITIES for item in modalities):
            parser.error('unknown MRI sequence in --modalities')
    if len(modalities) < 2 or len(set(modalities)) != len(modalities):
        parser.error('at least two unique MRI sequences are required')

    patient_visits, records = load_dataset(args.dataset, modalities, args.factor, args.patch)
    patients = sorted(patient_visits)
    if args.max_patients:
        patients = patients[:args.max_patients]
        records = [record for record in records if record[0] in set(patients)]
    if len(patients) < max(5, args.folds):
        parser.error('not enough complete patients for requested cross-validation')
    if args.folds < 2:
        parser.error('--folds must be at least 2')

    group_size = 64
    groups = tuple(tuple(range(i*group_size, (i+1)*group_size))
                   for i in range(len(modalities)))
    cue_channels = context_channels(len(modalities))
    total_squared_error = {name: np.zeros((len(modalities), 64), dtype=np.float64)
                           for name in ('mycelia', 'ridge')}
    total_patches = np.zeros(len(modalities), dtype=np.int64)
    fold_rows = []
    for fold in range(args.folds):
        train_ids, test_ids = pilot.patient_split(
            patients, args.seed, fold, args.folds)
        if train_ids & test_ids:
            raise AssertionError('patient overlap in outer fold')
        train = [record for record in records if record[0] in train_ids]
        test = [record for record in records if record[0] in test_ids]
        fingerprints = {}
        for record in records:
            patient, visit, _, _, _, shape, visual = record
            key = hashlib.sha256(
                str(shape).encode()+np.asarray(visual, dtype='<f4').tobytes()).hexdigest()
            fingerprints.setdefault(key, []).append(patient)
        cross_duplicates = [key for key, owners in fingerprints.items()
                            if len(set(owners)) > 1 and
                            any(owner in train_ids for owner in owners) and
                            any(owner in test_ids for owner in owners)]
        if cross_duplicates:
            raise SystemExit(f'fold {fold}: exact downsampled scan duplicates cross split')

        train_grids = [pilot.grid_for_visit(r[6], r[4], r[5]) for r in train]
        test_grids = [pilot.grid_for_visit(r[6], r[4], r[5]) for r in test]
        memory = VolumetricMycelium(VolumetricMemoryConfig(
            max_colonies=args.clusters, novelty_threshold=.08,
            learning_rate=.04, context_bandwidth=.18))
        memory.fit(train_grids, seed=args.seed+fold, fit_predictors=False)
        predictor = GroupedPredictiveMemory(
            memory, groups, context_scales=(1, 2),
            context_feature_channels=cue_channels).fit(train_grids)

        # Specialist choices see only inner held-out patients drawn from outer
        # training patients. The outer test fold never influences the gate.
        fold_args = argparse.Namespace(seed=args.seed, fold=fold, folds=args.folds,
                                       clusters=args.clusters, native_streaming=False)
        weights, calibration = pilot.calibrate_grouped_specialist_influence(
            train, train_ids, fold_args, groups,
            context_feature_channels=cue_channels, fit_predictors=False,
            target_memory=memory)

        ridge_models = []
        for group_index, group in enumerate(groups):
            examples = [predictor.masked_examples(grid, group_index)[:2]
                        for grid in train_grids]
            ridge_models.append(Ridge(alpha=10.0).fit(
                np.concatenate([item[0] for item in examples]),
                np.concatenate([item[1] for item in examples])))

        fold_patch_counts = np.zeros(len(modalities), dtype=np.int64)
        fold_model_error = np.zeros((len(modalities), 64), dtype=np.float64)
        fold_ridge_error = np.zeros((len(modalities), 64), dtype=np.float64)
        heldout_clusters, heldout_labels = [], []
        for record, grid in zip(test, test_grids):
            _, _, _, labels, coords, shape, _ = record
            assignments = memory.assignments(grid)[tuple(coords.T)]
            heldout_clusters.append(assignments)
            heldout_labels.append(labels)
            prediction = predictor.predict_volume(grid, local_weights=weights)['predictions']
            flat_indices = np.ravel_multi_index(coords.T, shape)
            for modality_index, group in enumerate(groups):
                visible = [index for index in range(grid.shape[-1]) if index not in group]
                sampled = grid.reshape(-1, grid.shape[-1])[flat_indices]
                valid = np.max(np.abs(sampled[:, visible]), axis=1) > memory.config.activity_threshold
                if not valid.any():
                    continue
                indices = flat_indices[valid]
                target = sampled[valid][:, list(group)]
                estimate = prediction[tuple(coords[valid].T)][:, list(group)]
                context = predictor.context_matrix(
                    grid, group, predictor.context_scales,
                    predictor.context_feature_channels)[indices]
                baseline = ridge_models[modality_index].predict(context)
                model_error = (estimate-target)**2
                ridge_error = (baseline-target)**2
                fold_model_error[modality_index] += model_error.sum(axis=0)
                fold_ridge_error[modality_index] += ridge_error.sum(axis=0)
                fold_patch_counts[modality_index] += int(valid.sum())

        labels = np.concatenate(heldout_labels)
        clusters = np.concatenate(heldout_clusters)
        nonbackground = labels > 0
        alignment = None
        if nonbackground.any():
            alignment = {
                'n_nonbackground_patches': int(nonbackground.sum()),
                'adjusted_rand_index': float(adjusted_rand_score(
                    labels[nonbackground], clusters[nonbackground])),
                'normalized_mutual_information': float(normalized_mutual_info_score(
                    labels[nonbackground], clusters[nonbackground])),
            }
        total_squared_error['mycelia'] += fold_model_error
        total_squared_error['ridge'] += fold_ridge_error
        total_patches += fold_patch_counts
        fold_rows.append({
            'fold': fold,
            'train_patients': sorted(train_ids), 'test_patients': sorted(test_ids),
            'train_visits': len(train), 'test_visits': len(test),
            'duplicate_downsampled_inputs_cross_split': 0,
            'patches_by_sequence': dict(zip(modalities, map(int, fold_patch_counts))),
            'selected_colony_weights': weights.tolist(),
            'specialist_calibration': calibration,
            'colony_mask_alignment_posthoc_only': alignment,
            'training_state': memory.validate(),
        })
        print(f"fold {fold+1}/{args.folds} complete: "
              f"{len(test_ids)} held-out patients; "
              f"{int(fold_patch_counts.sum())} sequence-patches", flush=True)

    metrics = {}
    for index, modality in enumerate(modalities):
        mycelia_mse = total_squared_error['mycelia'][index]/max(1, total_patches[index])
        ridge_mse = total_squared_error['ridge'][index]/max(1, total_patches[index])
        mycelia_voxel_mse = full_dct_projection_mse(mycelia_mse)
        ridge_voxel_mse = full_dct_projection_mse(ridge_mse)
        metrics[modality] = {
            'n_patches': int(total_patches[index]),
            'mycelia_coefficient_mse': mycelia_mse.tolist(),
            'ridge_coefficient_mse': ridge_mse.tolist(),
            'mycelia_coefficient_mse_ratio': float(mycelia_mse.mean()/ridge_mse.mean()),
            'mycelia_coefficients_better': int(np.sum(mycelia_mse < ridge_mse)),
            'ridge_coefficients_better': int(np.sum(ridge_mse < mycelia_mse)),
            'mycelia_full_patch_voxel_rmse': float(np.sqrt(mycelia_voxel_mse)),
            'ridge_full_patch_voxel_rmse': float(np.sqrt(ridge_voxel_mse)),
            'full_patch_voxel_mse_relative_change': float(
                (mycelia_voxel_mse/ridge_voxel_mse)-1.0),
        }
    output = {
        'protocol': 'patient-disjoint 5-fold unsupervised masked-sequence prediction',
        'visual_representation': 'all 64 signed orthonormal DCT coefficients of each 4x4x4 patch',
        'context_representation': 'DCT8 modes from neighboring 3D patches and visible sequences',
        'patch_resolution': 4, 'sampling_factor': args.factor,
        'modalities': list(modalities), 'seed': args.seed, 'folds': args.folds,
        'total_unique_patients': len(set(p for row in fold_rows for p in row['test_patients'])),
        'total_heldout_visits': sum(row['test_visits'] for row in fold_rows),
        'patient_overlap_across_heldout_folds': False,
        'duplicate_downsampled_inputs_cross_split': 0,
        'labels_used_for_learning_or_calibration': False,
        'label_alignment_is_posthoc_only': True,
        'metrics_by_sequence': metrics,
        'folds_detail': fold_rows,
        'interpretation': 'Full 4x4x4 patch intensity is reconstructed from all 64 modes; this is not a tumor diagnosis or proof of fungal equivalence.',
    }
    summary_path = args.output/'cross-validation-summary.json'
    summary_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({'summary': str(summary_path), 'metrics_by_sequence': metrics}, indent=2))


if __name__ == '__main__':
    main()
