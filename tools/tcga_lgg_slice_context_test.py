"""Patient-held-out 2.5D masked-patch transfer on ordered TCGA-LGG TIFF slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from tcga_lgg_patient_transfer_test import load_tiff_images, mask_rgb_patch, patient_folds
from mycelia.visual_memory import VisualConfig, VisualMycelium
from mycelia.volume_context import DirectionalVolumeContext, blend_predictions


def colony_grid(model, cue):
    scores = model.patch_scores(cue)
    if not scores.shape[1]:
        return np.full((model.config.size // model.config.patch_size,) * 2, -1, dtype=int)
    valid = np.isfinite(np.min(scores, axis=1))
    assignments = np.full(scores.shape[0], -1, dtype=int)
    assignments[valid] = np.argmin(scores[valid], axis=1)
    side = model.config.size // model.config.patch_size
    return assignments.reshape(side, side)


def fit_context(records, images, patient_ids, args, seed):
    indices = [i for i, row in enumerate(records) if row['patient'] in patient_ids]
    model = VisualMycelium(VisualConfig(
        size=args.size, patch_size=args.patch_size, max_colonies=args.max_colonies,
        max_episodes=10_000, detailed=False, multiscale_features=False,
        preprocessing='fixed', hierarchical_context=True))
    for i in np.random.default_rng(seed).permutation(indices):
        model.learn(images[int(i)])
    model.freeze()
    model.validate(verify_episode_templates=True)
    volumes = {}
    for i in indices:
        record = records[i]
        volumes.setdefault(record['patient'], {})[record['slice']] = colony_grid(model, images[i])
    memory = DirectionalVolumeContext().fit(volumes)
    train_patches = np.stack([model.patches(images[i]) for i in indices])
    return model, memory, train_patches.mean(axis=0)


def calibrate_route_weight(records, images, train_patients, args, fold_index):
    """Choose the route/local blend on a separate patient subset of outer train."""
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    inner_groups = patient_folds(sorted(train_patients), args.seed+70_001+fold_index,
                                 min(args.folds, len(train_patients)))
    validation_patients = inner_groups[0]
    inner_train = set(train_patients)-validation_patients
    if (inner_train & set(validation_patients) or
            inner_train | set(validation_patients) != set(train_patients)):
        raise AssertionError('inner calibration split must partition outer training patients')
    model, memory, training_mean = fit_context(
        records, images, inner_train, args, args.seed+80_001+fold_index)
    validation_indices = [i for i, row in enumerate(records)
                          if row['patient'] in validation_patients]
    volumes, source_by_slice, patch_cache = {}, {}, {}
    for i in validation_indices:
        record = records[i]
        key = (record['patient'], record['slice'])
        volumes.setdefault(record['patient'], {})[record['slice']] = colony_grid(model, images[i])
        source_by_slice[key] = i
        patch_cache[key] = model.patches(images[i])

    patch_side = args.size//args.patch_size
    # A deterministic spread of sites limits tuning cost while covering edges
    # and interior; held-out outer-test pixels never choose this weight.
    calibration_sites = sorted({0, patch_side-1, patch_side*(patch_side-1), patch_side**2-1,
                                patch_side**2//3})
    patient_errors = {patient: {weight: [] for weight in candidates}
                      for patient in validation_patients}
    for (patient, slice_id), image_index in source_by_slice.items():
        original = images[image_index]
        target_patches = patch_cache[(patient, slice_id)]
        for patch_index in calibration_sites:
            masked = mask_rgb_patch(original, patch_index, args.patch_size)
            assignments = colony_grid(model, masked)
            query = dict(volumes[patient])
            query[slice_id] = assignments
            y, x = divmod(patch_index, patch_side)
            route = memory.predict(query_volume_by_patient(query, patient),
                                   patient, slice_id, y, x, model)['predicted_patch']
            neighbors = []
            for dz, dy, dx in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                nz, ny, nx = slice_id+dz, y+dy, x+dx
                neighbor_index = source_by_slice.get((patient, nz))
                if neighbor_index is not None and 0 <= ny < patch_side and 0 <= nx < patch_side:
                    neighbors.append(patch_cache[(patient, nz)][ny*patch_side+nx])
            local = np.mean(neighbors, axis=0) if neighbors else training_mean[patch_index]
            target = target_patches[patch_index]
            for weight in candidates:
                estimate = blend_predictions(route, local, weight)
                patient_errors[patient][weight].append(float(np.mean((estimate-target)**2)))
    macro = {weight: float(np.mean([np.mean(patient_errors[p][weight])
                                    for p in validation_patients])) for weight in candidates}
    chosen = min(candidates, key=lambda weight: (macro[weight], weight))
    return chosen, {'inner_train_patients': sorted(inner_train),
                    'validation_patients': sorted(validation_patients),
                    'calibration_patch_sites': calibration_sites,
                    'patient_macro_mse_by_route_weight': {str(k): v for k, v in macro.items()},
                    'selected_route_weight': chosen,
                    'validation_patch_count': sum(len(v[chosen]) for v in patient_errors.values())}


def score_fold(records, images, train_patients, test_patients, args, fold_index):
    train_indices = [i for i, row in enumerate(records) if row['patient'] in train_patients]
    test_indices = [i for i, row in enumerate(records) if row['patient'] in test_patients]
    for key in ('source_sha256', 'cue_sha256'):
        train_hashes = {records[i][key] for i in train_indices}
        test_hashes = {records[i][key] for i in test_indices}
        if train_hashes & test_hashes:
            raise ValueError(f'exact {key} overlap between train and test patients')
    if set(train_patients) & set(test_patients):
        raise AssertionError('outer patient split must be disjoint')

    route_weight, calibration = calibrate_route_weight(
        records, images, train_patients, args, fold_index)
    if set(calibration['validation_patients']) & set(test_patients):
        raise AssertionError('outer test patient entered inner calibration')
    model, memory, training_mean = fit_context(
        records, images, train_patients, args, args.seed+fold_index)

    test_volumes, source_by_slice = {}, {}
    for i in test_indices:
        record = records[i]
        patient = record['patient']
        test_volumes.setdefault(patient, {})[record['slice']] = colony_grid(model, images[i])
        source_by_slice[(patient, record['slice'])] = i

    patch_side = args.size // args.patch_size
    patch_count = patch_side ** 2
    patient_errors = {patient: {name: [] for name in
                      ('slice_context', 'adaptive_blend', 'in_slice_hierarchy',
                       'training_mean', 'six_neighbor_mean')}
                      for patient in test_patients}
    source_counts = []
    patch_cache = {(records[i]['patient'], records[i]['slice']): model.patches(images[i])
                   for i in test_indices}
    rows = []
    for (patient, slice_id), image_index in source_by_slice.items():
        original = images[image_index]
        target_patches = patch_cache[(patient, slice_id)]
        visible_volume = test_volumes[patient]
        for patch_index, target in enumerate(target_patches):
            masked = mask_rgb_patch(original, patch_index, args.patch_size)
            assignments = colony_grid(model, masked)
            # Sub in assignments from the masked slice only. The held-out patch
            # is subsequently excluded by both prediction methods.
            query_volume = dict(visible_volume)
            query_volume[slice_id] = assignments
            y, x = divmod(patch_index, patch_side)
            slice_prediction = memory.predict(query_volume_by_patient(query_volume, patient),
                                              patient, slice_id, y, x, model)
            in_slice = model.predict_masked_patch_from_colonies(
                masked, patch_index, patch_assignments=assignments.reshape(-1), context_order=2)

            neighbor_patches = []
            for dz, dy, dx in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                nz, ny, nx = slice_id + dz, y + dy, x + dx
                if nz not in visible_volume or not 0 <= ny < patch_side or not 0 <= nx < patch_side:
                    continue
                neighbor_index = source_by_slice.get((patient, nz))
                if neighbor_index is None:
                    continue
                neighbor_patches.append(patch_cache[(patient, nz)][ny*patch_side+nx])
            local_prediction = np.mean(neighbor_patches, axis=0) if neighbor_patches \
                               else training_mean[patch_index]
            blended = blend_predictions(slice_prediction['predicted_patch'],
                                        local_prediction, route_weight)
            predictions = {
                'slice_context': slice_prediction['predicted_patch'],
                'adaptive_blend': blended,
                'in_slice_hierarchy': in_slice['predicted_patch'],
                'training_mean': training_mean[patch_index],
                'six_neighbor_mean': np.mean(neighbor_patches, axis=0) if neighbor_patches
                                    else training_mean[patch_index],
            }
            errors = {}
            for name, prediction in predictions.items():
                # No-route cases use the training-mean fallback and are counted
                # explicitly; this gives the context system full test coverage.
                if prediction is None:
                    prediction = training_mean[patch_index]
                errors[name] = float(np.mean((prediction-target)**2))
                patient_errors[patient][name].append(errors[name])
            source_counts.append(slice_prediction['context_sources'])
            rows.append({'file': records[image_index]['file'], 'patient': patient,
                         'slice': slice_id, 'patch': patch_index,
                         **{f'{name}_mse': value for name, value in errors.items()},
                         'slice_context_sources': slice_prediction['context_sources'],
                         'slice_context_routes': len(slice_prediction['candidate_probabilities']),
                         'selected_route_weight': route_weight})
    return {'patient_errors': patient_errors, 'rows': rows,
            'train_patients': sorted(train_patients), 'test_patients': sorted(test_patients),
            'train_images': len(train_indices), 'test_images': len(test_indices),
            'slice_context_route_count': memory.route_count,
            'slice_context_update_count': memory.updates,
            'mean_context_sources': float(np.mean(source_counts)),
            'context_patch_coverage': float(np.mean(np.asarray(source_counts) > 0)),
            'selected_route_weight': route_weight,
            'inner_calibration': calibration,
            'learning_summary': model.summary()}


def query_volume_by_patient(slices, patient):
    """Adapt the single-patient test-slice mapping to the memory API."""
    return {patient: slices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=8)
    parser.add_argument('--max-colonies', type=int, default=128)
    args = parser.parse_args()
    if args.size < 8 or args.size % args.patch_size or args.patch_size < 3:
        parser.error('size must be divisible by patch-size, and patch-size must be >=3')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, images = load_tiff_images(args.dataset, args.size)
    patients = sorted({row['patient'] for row in records})
    groups = patient_folds(patients, args.seed, args.folds)
    folds = [score_fold(records, images, set(patients)-test, test, args, i)
             for i, test in enumerate(groups)]

    patient_errors = {}
    for fold in folds:
        patient_errors.update(fold['patient_errors'])
    names = ('slice_context', 'adaptive_blend', 'in_slice_hierarchy',
             'training_mean', 'six_neighbor_mean')
    per_patient_mse = {name: np.asarray([np.mean(patient_errors[p][name]) for p in sorted(patient_errors)])
                       for name in names}
    overall = {name: {'patient_macro_mse': float(values.mean()),
                      'patient_macro_rmse': float(np.sqrt(values.mean()))}
               for name, values in per_patient_mse.items()}
    rng = np.random.default_rng(args.seed+731)
    def compare(candidate, baseline):
        delta = per_patient_mse[candidate]-per_patient_mse[baseline]
        samples = rng.integers(0, len(delta), size=(20_000, len(delta)))
        return {'mean_patient_mse_difference': float(delta.mean()),
                'patient_bootstrap_95_percent_interval': [float(v) for v in
                    np.quantile(delta[samples].mean(axis=1), [.025,.975])],
                'fraction_patients_candidate_lower_mse': float(np.mean(delta < 0))}
    all_rows = [row for fold in folds for row in fold['rows']]
    report = {
        'scope': 'Patient-held-out, label-free, ordered-slice context prediction. This is 2.5D slice-index adjacency, not validated physical 3D geometry.',
        'dataset': {'images': len(records), 'patients': len(patients),
                    'slices_per_patient': {p: sum(r['patient']==p for r in records) for p in patients}},
        'protocol': 'Five-fold patient-grouped cross-validation. Within each outer training fold, hold out a subset of training patients to select a route/local-evidence blend weight by masked-patch MSE; refit on all outer training patients. At test time mask one complete patch before matching, then evaluate the selected blend and fixed baselines. Diagnosis labels and masks are not used for learning or selection.',
        'slice_adjacency': 'Only filename slice indices exactly one apart are connected; no geometric spacing or orientation metadata is available.',
        'exact_source_overlap': 0, 'encoded_cue_overlap': 0,
        'labels_used_for_learning_or_selection': False, 'tumor_masks_used': False,
        'patient_overlap': sum(bool(set(f['train_patients']) & set(f['test_patients'])) for f in folds),
        'patient_macro_metrics': overall,
        'slice_context_vs_in_slice_hierarchy': compare('slice_context','in_slice_hierarchy'),
        'slice_context_vs_training_mean': compare('slice_context','training_mean'),
        'slice_context_vs_six_neighbor_mean': compare('slice_context','six_neighbor_mean'),
        'adaptive_blend_vs_in_slice_hierarchy': compare('adaptive_blend','in_slice_hierarchy'),
        'adaptive_blend_vs_training_mean': compare('adaptive_blend','training_mean'),
        'adaptive_blend_vs_six_neighbor_mean': compare('adaptive_blend','six_neighbor_mean'),
        'selected_route_weights_by_outer_fold': [f['selected_route_weight'] for f in folds],
        'slice_context_patch_coverage': float(np.mean([f['context_patch_coverage'] for f in folds])),
        'mean_context_sources': float(np.mean([f['mean_context_sources'] for f in folds])),
        'folds': [{k:v for k,v in fold.items() if k not in {'rows','patient_errors'}} for fold in folds],
        'prediction_rows': len(all_rows),
        'limitations': ['Only 10 patients are locally available; estimates are exploratory.',
                        'Slice-index neighbors do not prove spatially adjacent 3D anatomy; acquisition geometry is missing.',
                        'The source is 2D RGB-encoded TIFFs, not original DICOM/NIfTI studies.',
                        'No diagnosis labels or masks are used, so this is not a tumor diagnosis test.',
                        'No external-site validation or biological equivalence is established.'],
        'elapsed_seconds': time.monotonic()-started,
    }
    (args.output/'slice-context-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    import csv
    with (args.output/'slice-context-predictions.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps({'dataset':report['dataset'],'patch_predictions':len(all_rows),
                      'metrics':overall,
                      'context_vs_2d':report['slice_context_vs_in_slice_hierarchy'],
                      'context_vs_mean':report['slice_context_vs_training_mean'],
                      'adaptive_blend_vs_neighbor_mean':report['adaptive_blend_vs_six_neighbor_mean'],
                      'selected_route_weights':report['selected_route_weights_by_outer_fold'],
                      'coverage':report['slice_context_patch_coverage'],
                      'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__ == '__main__':
    main()
