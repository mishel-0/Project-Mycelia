"""Evaluate label-free masked-patch transfer against simple visual baselines."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mycelia.visual_memory import VisualConfig, VisualMycelium, mask_gray_patch, visual_cues
from unlabeled_visual_test import fit, load_inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preprocessing', choices=('fixed', 'foreground'), default='fixed',
                        help='fixed uses no image-dependent crop or contrast statistics; foreground reproduces the earlier caveated path')
    parser.add_argument('--train-images', type=int, default=600)
    parser.add_argument('--validation-images', type=int, default=200)
    parser.add_argument('--validation-partition', choices=('Training', 'Testing'), default='Testing')
    parser.add_argument('--exclude-validation-csv', type=Path,
                        help='CSV of previously evaluated image paths to keep out of this validation sample')
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--max-colonies', type=int, default=256)
    parser.add_argument('--novelty-threshold', type=float, default=.012,
                        help='local patch mismatch required to recruit a specialist')
    parser.add_argument('--motif-merge-threshold', type=float, default=0.0,
                        help='opt-in symmetric prototype mismatch limit for merging specialists at capacity')
    parser.add_argument('--multiscale-cues', action=argparse.BooleanOptionalAction, default=False,
                        help='combine fine and broad-range edge/texture cues before label-free learning')
    parser.add_argument('--predictive-feedback', action=argparse.BooleanOptionalAction, default=False,
                        help='use pre-learning context prediction error to modulate local receptor updates')
    parser.add_argument('--hierarchical-context', action=argparse.BooleanOptionalAction, default=False,
                        help='learn and evaluate pair-conditioned three-patch motifs')
    args = parser.parse_args()
    if min(args.train_images, args.validation_images, args.max_colonies) < 1:
        parser.error('image counts and max-colonies must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    records, gray32, _ = load_inputs(args.dataset, preprocessing=args.preprocessing)
    # Diagnosis folder names remain in the audit manifest, but are never read
    # as model inputs or targets. The default validation set comes from the
    # separate Testing partition, avoiding tuning on a Training holdout.
    training = [i for i, row in enumerate(records) if Path(row['file']).parts[0] == 'Training']
    unique, seen_source, seen_cue = [], set(), set()
    for i in training:
        row = records[i]
        if row['source_sha256'] in seen_source or row['cue32_sha256'] in seen_cue:
            continue
        seen_source.add(row['source_sha256']); seen_cue.add(row['cue32_sha256']); unique.append(i)
    if len(unique) < args.train_images:
        raise ValueError('not enough unique Training images for requested training subset')
    rng = np.random.default_rng(args.seed)
    train_indices = rng.permutation(np.asarray(unique, dtype=int))[:args.train_images]
    train_source = {records[i]['source_sha256'] for i in train_indices}
    train_cues = {records[i]['cue32_sha256'] for i in train_indices}
    excluded_validation = set()
    if args.exclude_validation_csv:
        with args.exclude_validation_csv.open(newline='') as stream:
            excluded_validation = {row['image'] for row in csv.DictReader(stream) if row.get('image')}
    validation_pool, val_source, val_cues = [], set(), set()
    for i, row in enumerate(records):
        if Path(row['file']).parts[0] != args.validation_partition:
            continue
        if row['file'] in excluded_validation:
            continue
        if row['source_sha256'] in train_source or row['cue32_sha256'] in train_cues:
            continue
        if row['source_sha256'] in val_source or row['cue32_sha256'] in val_cues:
            continue
        validation_pool.append(i)
        val_source.add(row['source_sha256']); val_cues.add(row['cue32_sha256'])
    if len(validation_pool) < args.validation_images:
        raise ValueError(f'not enough unique {args.validation_partition} images after duplicate exclusion')
    validation_indices = rng.permutation(np.asarray(validation_pool, dtype=int))[:args.validation_images]
    if train_source & {records[i]['source_sha256'] for i in validation_indices}:
        raise AssertionError('source duplicate crossed the split')
    if train_cues & {records[i]['cue32_sha256'] for i in validation_indices}:
        raise AssertionError('encoded duplicate crossed the split')

    cues = np.stack([visual_cues(gray, multiscale=args.multiscale_cues) for gray in gray32])
    model, learning = fit(cues, train_indices,
                          VisualConfig(max_colonies=args.max_colonies,
                                       novelty_threshold=args.novelty_threshold,
                                       motif_merge_threshold=args.motif_merge_threshold,
                                       multiscale_features=args.multiscale_cues,
                                       preprocessing=args.preprocessing,
                                       hierarchical_context=args.hierarchical_context,
                                       predictive_feedback=args.predictive_feedback), 1,
                          args.seed, args.output, 'masked-patch-training')
    model.freeze()
    checkpoint_path = args.output/'unlabeled-visual-context-memory.json'
    model.save(checkpoint_path)
    restored = VisualMycelium.load(checkpoint_path)
    if restored.state_dict() != model.state_dict():
        raise RuntimeError('predictive-context checkpoint round-trip changed model state')
    model = restored
    patch_grid = model.patches(cues[train_indices[0]])
    side = int(round(np.sqrt(len(patch_grid))))
    train_patches = np.stack([model.patches(cues[i]) for i in train_indices])
    global_mean = train_patches.mean(axis=0)
    episode_squared, context_fallback_squared = [], []
    context_covered_squared, mean_squared, neighbor_squared = [], [], []
    hierarchical_covered_squared, hierarchical_fallback_squared = [], []
    hierarchical_motif_predictions = hierarchical_fallbacks = 0
    context_predictions = 0
    rows = []
    for image_index in validation_indices:
        image_patches = model.patches(cues[image_index])
        per_episode, per_context, per_context_fallback = [], [], []
        per_hierarchical_fallback = []
        per_mean, per_neighbor = [], []
        for patch_index, target in enumerate(image_patches):
            masked_gray = mask_gray_patch(gray32[image_index], patch_index,
                                          model.config.patch_size)
            masked_cue = visual_cues(masked_gray, multiscale=args.multiscale_cues)
            patch_scores = model.patch_scores(masked_cue)
            visible_assignments = np.full(len(image_patches), -1, dtype=int)
            if patch_scores.shape[1]:
                valid = np.isfinite(np.min(patch_scores, axis=1))
                visible_assignments[valid] = np.argmin(patch_scores[valid], axis=1)
            prediction = model.predict_masked_patch(masked_cue, patch_index)
            model_patch = prediction['predicted_patch']
            local_prediction = model.predict_masked_patch_from_colonies(
                masked_cue, patch_index, patch_assignments=visible_assignments)
            hierarchical_prediction = (model.predict_masked_patch_from_colonies(
                masked_cue, patch_index, patch_assignments=visible_assignments, context_order=2)
                if args.hierarchical_context else None)
            y, x = divmod(patch_index, side)
            adjacent = [image_patches[yy*side+xx] for yy, xx in
                        ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if 0 <= yy < side and 0 <= xx < side]
            neighbor_patch = np.mean(adjacent, axis=0)
            episode_mse = float(np.mean((model_patch[..., 0]-target[..., 0])**2))
            mean_mse = float(np.mean((global_mean[patch_index, ..., 0]-target[..., 0])**2))
            neighbor_mse = float(np.mean((neighbor_patch[..., 0]-target[..., 0])**2))
            if local_prediction['predicted_patch'] is None:
                context_mse = None
                fallback_mse = mean_mse
            else:
                context_mse = float(np.mean((local_prediction['predicted_patch'][..., 0]-target[..., 0])**2))
                fallback_mse = context_mse
                per_context.append(context_mse)
                context_predictions += 1
            if hierarchical_prediction is not None:
                if hierarchical_prediction['predicted_patch'] is None:
                    hierarchical_mse = None
                    hierarchical_fallback_mse = mean_mse
                    hierarchical_fallbacks += 1
                else:
                    hierarchical_mse = float(np.mean((hierarchical_prediction['predicted_patch'][..., 0]-target[..., 0])**2))
                    hierarchical_fallback_mse = hierarchical_mse
                    hierarchical_covered_squared.append(hierarchical_mse)
                    if hierarchical_prediction.get('context_order_used') == 2:
                        hierarchical_motif_predictions += 1
                per_hierarchical_fallback.append(hierarchical_fallback_mse)
            per_episode.append(episode_mse); per_context_fallback.append(fallback_mse)
            per_mean.append(mean_mse); per_neighbor.append(neighbor_mse)
            if hierarchical_prediction is None:
                rows.append((records[image_index]['file'], patch_index, prediction['candidate_identity'],
                             prediction['context_rmse'], episode_mse, context_mse, fallback_mse,
                             mean_mse, neighbor_mse, local_prediction['context_sources']))
            else:
                rows.append((records[image_index]['file'], patch_index, prediction['candidate_identity'],
                             prediction['context_rmse'], episode_mse, context_mse, fallback_mse,
                             hierarchical_mse, hierarchical_prediction.get('context_order_used'),
                             hierarchical_prediction['context_sources'], hierarchical_fallback_mse,
                             mean_mse, neighbor_mse, local_prediction['context_sources']))
        episode_squared.append(float(np.mean(per_episode)))
        context_covered_squared.extend(per_context)
        context_fallback_squared.append(float(np.mean(per_context_fallback)))
        if hierarchical_prediction is not None:
            hierarchical_fallback_squared.append(float(np.mean(per_hierarchical_fallback)))
        mean_squared.append(float(np.mean(per_mean)))
        neighbor_squared.append(float(np.mean(per_neighbor)))

    def metrics(values, baseline):
        mse = float(np.mean(values))
        return dict(mse=mse, rmse=float(np.sqrt(mse)), relative_mse_reduction_vs_baseline=1-mse/max(float(np.mean(baseline)),1e-12))

    context_covered_mse = float(np.mean(context_covered_squared)) if context_covered_squared else None
    paired_delta = np.asarray(context_fallback_squared)-np.asarray(mean_squared)
    bootstrap_rng = np.random.default_rng(args.seed+1)
    bootstrap_indices = bootstrap_rng.integers(0, len(paired_delta), size=(20_000, len(paired_delta)))
    paired_bootstrap = paired_delta[bootstrap_indices].mean(axis=1)
    paired_summary = {
        'images': len(paired_delta),
        'mean_context_minus_spatial_mean_mse': float(np.mean(paired_delta)),
        'paired_image_bootstrap_95_percent_interval': [float(v) for v in np.quantile(paired_bootstrap, [.025, .975])],
        'fraction_images_context_lower_mse': float(np.mean(paired_delta < 0)),
        'bootstrap_resamples': len(paired_bootstrap),
        'unit': 'image; patient-level independence is unknown',
    }
    hierarchical_paired_summary = None
    if args.hierarchical_context:
        hierarchical_delta = np.asarray(hierarchical_fallback_squared)-np.asarray(mean_squared)
        hierarchical_bootstrap = hierarchical_delta[bootstrap_indices].mean(axis=1)
        hierarchical_vs_first_order = (np.asarray(hierarchical_fallback_squared)-
                                       np.asarray(context_fallback_squared))
        hierarchical_vs_first_bootstrap = hierarchical_vs_first_order[bootstrap_indices].mean(axis=1)
        hierarchical_paired_summary = {
            'images': len(hierarchical_delta),
            'mean_hierarchical_minus_spatial_mean_mse': float(np.mean(hierarchical_delta)),
            'paired_image_bootstrap_95_percent_interval': [float(v) for v in np.quantile(hierarchical_bootstrap, [.025, .975])],
            'fraction_images_hierarchical_lower_mse': float(np.mean(hierarchical_delta < 0)),
            'mean_hierarchical_minus_first_order_mse': float(np.mean(hierarchical_vs_first_order)),
            'paired_hierarchical_vs_first_order_95_percent_interval': [
                float(v) for v in np.quantile(hierarchical_vs_first_bootstrap, [.025, .975])],
            'fraction_images_hierarchical_lower_than_first_order': float(np.mean(hierarchical_vs_first_order < 0)),
            'unit': 'image; patient-level independence is unknown',
        }

    report = {
        'protocol': f'Train on unique images from Training only, with no class labels supplied; evaluate on disjoint unique {args.validation_partition} images after source-pixel and cue-encoding duplicate checks. Exclude {len(excluded_validation)} previously evaluated image paths when the optional exclusion CSV is provided. For each held-out image and target patch, replace target pixels with the mean of visible pixels before computing edge/texture cues, then predict from the remaining visible grayscale patch context using the nearest stored episode and learned colony context. Compare with a training-only spatial global-mean patch and same-image adjacent-patch baseline.',
        'target_masking': 'target patch replaced by visible-image mean before feature extraction and matching',
            'preprocessing': args.preprocessing,
            'preprocessing_caveat': ('none: fixed full-frame resize and source grayscale scale are independent of image content; target masking then precedes cue extraction' if args.preprocessing == 'fixed' else 'per-image foreground crop and contrast normalization occur before target masking; local cue leakage is blocked, but global intensity statistics may still depend on the complete scan'),
        'scope': 'Unsupervised image-level transfer diagnostic. Dataset patient identifiers are unavailable; this does not test patient-disjoint generalization, diagnosis, or biological realism.',
        'counts': {'unique_training_available': len(unique), 'train_images': len(train_indices),
                   'validation_images': len(validation_indices), 'masked_patch_predictions': len(rows),
                   'previously_evaluated_paths_excluded': sum(row['file'] in excluded_validation for row in records),
                   'source_overlap': len(train_source & {records[i]['source_sha256'] for i in validation_indices}),
                   'encoded_overlap': len(train_cues & {records[i]['cue32_sha256'] for i in validation_indices})},
        'learning': learning,
        'model_settings': {'max_colonies': args.max_colonies,
                           'novelty_threshold': args.novelty_threshold,
                           'motif_merge_threshold': args.motif_merge_threshold,
                           'multiscale_features': args.multiscale_cues,
                           'hierarchical_context': args.hierarchical_context},
        'predictive_feedback_enabled': args.predictive_feedback,
        'checkpoint': {'file': checkpoint_path.name, 'round_trip_exact': True,
                       'schema': model.state_dict()['schema']},
        'gray_intensity_patch_metrics': {
            'episodic_context_prediction': metrics(episode_squared, mean_squared),
            'learned_colony_context_prediction': {
                'predictions_with_learned_context': context_predictions,
                'coverage': context_predictions/max(len(rows), 1),
                'covered_mse': context_covered_mse,
                'covered_rmse': float(np.sqrt(context_covered_mse)) if context_covered_mse is not None else None,
                'mse_with_spatial_mean_fallback': float(np.mean(context_fallback_squared)),
                'rmse_with_spatial_mean_fallback': float(np.sqrt(np.mean(context_fallback_squared))),
            },
            'training_spatial_global_mean': metrics(mean_squared, mean_squared),
            'same_image_adjacent_patch_mean': metrics(neighbor_squared, mean_squared),
        },
        'paired_image_statistics': paired_summary,
        'patch_positions_tested': 'all positions, each patch held out independently',
        'labels_used_for_learning': False,
        'labels_used_for_selection_or_metrics': False,
        'elapsed_seconds': time.monotonic()-started,
        'interpretation': 'The learned colony-context predictor uses local directional associations between patch specialists, not the episodic whole-image templates. Compare coverage and held-out error against the episodic and simple baselines. This remains a toy image-level diagnostic, not semantic understanding or clinical performance.',
    }
    if args.hierarchical_context:
        hierarchical_mse = float(np.mean(hierarchical_covered_squared)) if hierarchical_covered_squared else None
        report['gray_intensity_patch_metrics']['learned_hierarchical_context_prediction'] = {
            'predictions_with_learned_context': len(rows)-hierarchical_fallbacks,
            'motif_conditioned_predictions': hierarchical_motif_predictions,
            'motif_conditioned_fraction': hierarchical_motif_predictions/max(len(rows), 1),
            'first_order_backoff_predictions': max(0, len(rows)-hierarchical_motif_predictions-hierarchical_fallbacks),
            'no_context_fallback_predictions': hierarchical_fallbacks,
            'covered_mse': hierarchical_mse,
            'covered_rmse': float(np.sqrt(hierarchical_mse)) if hierarchical_mse is not None else None,
            'mse_with_spatial_mean_fallback': float(np.mean(hierarchical_fallback_squared)),
            'rmse_with_spatial_mean_fallback': float(np.sqrt(np.mean(hierarchical_fallback_squared))),
        }
        report['paired_image_statistics']['learned_hierarchical_context_prediction'] = hierarchical_paired_summary
    write_json(args.output/'masked-prediction-summary.json', report)
    with (args.output/'masked-predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        if args.hierarchical_context:
            writer.writerow(['image', 'masked_patch', 'retrieved_episode', 'visible_context_rmse',
                             'episode_gray_mse', 'learned_colony_context_gray_mse',
                             'context_with_mean_fallback_gray_mse', 'hierarchical_context_gray_mse',
                             'context_order_used', 'hierarchical_context_sources',
                             'hierarchical_with_mean_fallback_gray_mse', 'training_global_mean_gray_mse',
                             'adjacent_patch_gray_mse', 'context_sources'])
        else:
            writer.writerow(['image', 'masked_patch', 'retrieved_episode', 'visible_context_rmse',
                             'episode_gray_mse', 'learned_colony_context_gray_mse',
                             'context_with_mean_fallback_gray_mse', 'training_global_mean_gray_mse',
                             'adjacent_patch_gray_mse', 'context_sources'])
        writer.writerows(rows)
    print(json.dumps({'counts': report['counts'], 'metrics': report['gray_intensity_patch_metrics'],
                      'elapsed_seconds': report['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
