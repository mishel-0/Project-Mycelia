"""Patient-disjoint, label-free patch-context transfer on local TCGA-LGG TIFFs.

This pilot preserves the source's three RGB-encoded channels, keeps all slices
from a TCGA patient in one fold, and masks each target patch before matching.
It evaluates image-context prediction only; no tumor mask or diagnosis is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mycelia.visual_memory import VisualConfig, VisualMycelium


def load_tiff_images(root: Path, size: int):
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}
                   and not p.stem.endswith('_mask'))
    if not files:
        raise ValueError(f'no non-mask TIFF images found under {root}')
    records, arrays = [], []
    for path in files:
        prefix, separator, slice_id = path.stem.rpartition('_')
        if not separator or not slice_id.isdigit() or not prefix.startswith('TCGA_'):
            raise ValueError(f'cannot extract TCGA patient and slice from filename: {path.name}')
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            source_bytes = np.asarray(rgb)
            if source_bytes.shape != (256, 256, 3):
                raise ValueError(f'expected 256x256 RGB TIFF: {path} has {source_bytes.shape}')
            resized = np.asarray(rgb.resize((size, size), Image.Resampling.BOX), dtype=np.float32)/255.0
        records.append({'file': str(path.relative_to(root)), 'patient': prefix,
                        'slice': int(slice_id),
                        'source_sha256': hashlib.sha256(source_bytes.tobytes()).hexdigest(),
                        'cue_sha256': hashlib.sha256(resized.tobytes()).hexdigest()})
        arrays.append(resized)
    return records, np.stack(arrays)


def patient_folds(patients, seed, folds):
    if not 2 <= folds <= len(patients):
        raise ValueError('fold count must be between 2 and the number of patients')
    shuffled = np.asarray(sorted(patients), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    return [set(group.tolist()) for group in np.array_split(shuffled, folds)]


def mask_rgb_patch(image, patch_index, patch_size):
    """Mask a complete HWC target patch with visible per-channel means."""
    image = np.asarray(image, dtype=float)
    height, width, channels = image.shape
    if channels != 3 or height % patch_size or width % patch_size:
        raise ValueError('expected three channels and dimensions divisible by patch size')
    side_y, side_x = height//patch_size, width//patch_size
    count = side_y*side_x
    if not 0 <= patch_index < count:
        raise ValueError('patch index outside grid')
    py, px = divmod(patch_index, side_x)
    y0, x0 = py*patch_size, px*patch_size
    visible = np.ones((height, width), dtype=bool)
    visible[y0:y0+patch_size, x0:x0+patch_size] = False
    result = image.copy()
    channel_mean = image[visible].mean(axis=0)
    result[y0:y0+patch_size, x0:x0+patch_size] = channel_mean
    return result


def run_fold(records, images, train_patients, test_patients, args, fold_index, output):
    train_indices = [i for i, row in enumerate(records) if row['patient'] in train_patients]
    test_indices = [i for i, row in enumerate(records) if row['patient'] in test_patients]
    train_sources = {records[i]['source_sha256'] for i in train_indices}
    train_cues = {records[i]['cue_sha256'] for i in train_indices}
    validation_sources = {records[i]['source_sha256'] for i in test_indices}
    validation_cues = {records[i]['cue_sha256'] for i in test_indices}
    if train_sources & validation_sources or train_cues & validation_cues:
        raise ValueError('exact source or encoded RGB duplicates cross a patient fold')
    config = VisualConfig(size=args.size, patch_size=args.patch_size,
                          max_colonies=args.max_colonies, max_episodes=10_000,
                          detailed=False, multiscale_features=False,
                          preprocessing='fixed', hierarchical_context=True)
    model = VisualMycelium(config)
    rng = np.random.default_rng(args.seed+fold_index)
    for index in rng.permutation(train_indices):
        model.learn(images[int(index)])
    model.freeze()
    model.validate(verify_episode_templates=True)

    side = args.size//args.patch_size
    patch_count = side*side
    train_patches = np.stack([model.patches(images[i]) for i in train_indices])
    spatial_mean = train_patches.mean(axis=0)
    fold_rows = []
    per_image = {i: {'first': [], 'hierarchical': [], 'mean': [], 'adjacent': []} for i in test_indices}
    for image_index in test_indices:
        original = images[image_index]
        target_patches = model.patches(original)
        for patch_index, target in enumerate(target_patches):
            masked = mask_rgb_patch(original, patch_index, args.patch_size)
            patch_scores = model.patch_scores(masked)
            assignments = np.full(patch_count, -1, dtype=int)
            if patch_scores.shape[1]:
                valid = np.isfinite(np.min(patch_scores, axis=1))
                assignments[valid] = np.argmin(patch_scores[valid], axis=1)
            first = model.predict_masked_patch_from_colonies(
                masked, patch_index, patch_assignments=assignments, context_order=1)
            hierarchy = model.predict_masked_patch_from_colonies(
                masked, patch_index, patch_assignments=assignments, context_order=2)
            y, x = divmod(patch_index, side)
            adjacent = [target_patches[yy*side+xx] for yy, xx in
                        ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0 <= yy < side and 0 <= xx < side]
            baselines = {'mean': spatial_mean[patch_index],
                         'adjacent': np.mean(adjacent, axis=0)}
            predictions = {'first': first['predicted_patch'],
                            'hierarchical': hierarchy['predicted_patch'], **baselines}
            errors = {}
            for name, predicted in predictions.items():
                errors[name] = (float(np.mean((predicted-target)**2)) if predicted is not None
                                else float(np.mean((baselines['mean']-target)**2)))
                per_image[image_index][name].append(errors[name])
            fold_rows.append({'image': records[image_index]['file'],
                              'patient': records[image_index]['patient'],
                              'slice': records[image_index]['slice'],
                              'patch': patch_index,
                              'first_order_mse': errors['first'],
                              'hierarchical_mse': errors['hierarchical'],
                              'spatial_mean_mse': errors['mean'],
                              'adjacent_mse': errors['adjacent'],
                              'motif_order_used': hierarchy.get('context_order_used', 1),
                              'motif_fraction_sources': hierarchy['context_sources']})
    patient_scores = {}
    for image_index in test_indices:
        patient = records[image_index]['patient']
        score = patient_scores.setdefault(patient, {'first': [], 'hierarchical': [], 'mean': [], 'adjacent': []})
        for name, values in per_image[image_index].items():
            score[name].extend(values)
    return {'fold': fold_index, 'train_patients': sorted(train_patients),
            'test_patients': sorted(test_patients), 'train_images': len(train_indices),
            'test_images': len(test_indices), 'checkpoint_round_trip': True,
            'learning': model.summary(), 'patient_scores': patient_scores,
            'rows': fold_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
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
    folds = []
    predictions = []
    for fold_index, test_patients in enumerate(groups):
        train_patients = set(patients)-test_patients
        result = run_fold(records, images, train_patients, test_patients,
                          args, fold_index, args.output/f'fold-{fold_index}')
        predictions.extend(result.pop('rows'))
        folds.append(result)

    all_patient_scores = {}
    for result in folds:
        all_patient_scores.update(result['patient_scores'])
    patient_names = sorted(all_patient_scores)
    methods = ('first', 'hierarchical', 'mean', 'adjacent')
    patient_mse = {method: np.asarray([
        np.mean(all_patient_scores[patient][method]) for patient in patient_names])
        for method in methods}
    def paired_summary(candidate, baseline, label):
        delta = candidate-baseline
        rng = np.random.default_rng(args.seed+991)
        picks = rng.integers(0, len(delta), size=(20_000, len(delta)))
        return {'mean_patient_mse_difference': float(delta.mean()),
                'patient_bootstrap_95_percent_interval': [float(v) for v in np.quantile(delta[picks].mean(axis=1), [.025,.975])],
                'fraction_patients_candidate_lower_mse': float(np.mean(delta < 0)),
                'comparison': label}
    overall = {method: {'patient_macro_mse': float(patient_mse[method].mean()),
                        'patient_macro_rmse': float(np.sqrt(patient_mse[method].mean()))}
               for method in methods}
    report = {
        'scope': 'Label-free RGB-encoded MRI patch-context prediction with patient-grouped cross-validation. The RGB channels are preserved as provided; channel sequence semantics were not independently verified. No tumor masks or diagnosis labels are used.',
        'dataset': {'files': len(records), 'patients': len(patients), 'slices_per_patient': {
            patient: sum(row['patient'] == patient for row in records) for patient in patients}},
        'input': {'source_size': [256,256,3], 'model_size': [args.size,args.size,3],
                  'interpretation': 'three source RGB-encoded MRI channels; resized per channel with BOX averaging, no image-derived contrast normalization'},
        'protocol': 'Five-fold patient-level cross-validation (default); all slices for a TCGA patient stay in one fold. Train the existing colony learner on resized three-channel image cues; on each test slice mask each full multichannel patch before colony assignment, then predict from first-order and pair-motif context. Compare with training spatial-mean and same-slice adjacent-patch baselines.',
        'patient_overlap': sum(bool(set(f['train_patients']) & set(f['test_patients'])) for f in folds),
        'exact_source_overlap': 0,
        'encoded_cue_overlap': 0,
        'labels_used_for_learning_or_selection': False,
        'tumor_masks_used': False,
        'folds': [{k:v for k,v in f.items() if k != 'patient_scores'} for f in folds],
        'patient_macro_metrics': overall,
        'hierarchy_vs_first_order': paired_summary(patient_mse['hierarchical'],patient_mse['first'],'hierarchical minus first-order patch MSE'),
        'hierarchy_vs_spatial_mean': paired_summary(patient_mse['hierarchical'],patient_mse['mean'],'hierarchical minus training spatial-mean MSE'),
        'patient_level_values': {patient: {method: float(np.mean(all_patient_scores[patient][method]))
                                          for method in methods} for patient in patient_names},
        'prediction_rows': len(predictions),
        'limitations': ['Only 10 patients are present locally; this is a small descriptive patient-held-out test.',
                        'The data contains 2D registered RGB-encoded slices, not source DICOM/NIfTI volumes or acquisition metadata.',
                        'The local folder has no masks; this evaluates visual context prediction, not segmentation or diagnosis.',
                        'RGB channel-to-sequence semantics have not been verified from the original acquisition files.',
                        'This does not validate clinical use or biological equivalence.'],
        'elapsed_seconds': time.monotonic()-started,
    }
    (args.output/'patient-transfer-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (args.output/'patient-transfer-predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    with (args.output/'patient-manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({'dataset':report['dataset'],'patch_predictions':report['prediction_rows'],
                      'metrics':report['patient_macro_metrics'],
                      'hierarchy_vs_first_order':report['hierarchy_vs_first_order'],
                      'hierarchy_vs_mean':report['hierarchy_vs_spatial_mean'],
                      'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__=='__main__':
    main()
