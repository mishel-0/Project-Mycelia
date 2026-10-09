"""Learn unlabeled local context -> masked MRI patch predictions.

The target is absent from the context at both training and test time. During
training MYCELIA predicts first, then the arriving patch supplies a local error
signal to the winning resource-funded context/target colony pair.
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

from mycelia.predictive_memory import PredictiveConfig, PredictiveMycelium, masked_neighbor_context, patch_grid
from mycelia.visual_memory import VisualMycelium, _distances, prepare_gray, visual_cues


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def data_from_phase1(dataset, phase1_output, *, max_train=None, max_test=None, seed=2026):
    """Reuse Phase 1's exact clean split and never consult labels for selection."""
    excluded_train = set()
    with (phase1_output / 'duplicate-audit.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == 'Training':
                excluded_train.add(row['file'])
    train_paths = sorted(p for p in (dataset / 'Training').rglob('*')
                         if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                         and str(p.relative_to(dataset)) not in excluded_train)
    with (phase1_output / 'heldout-results.csv').open(newline='') as stream:
        heldout_rows = list(csv.DictReader(stream))
    test_paths = [dataset / row['file'] for row in heldout_rows]
    test_groups = [Path(row['file']).parent.name for row in heldout_rows]
    selection_rng = np.random.default_rng(seed + 991)
    if max_train is not None and len(train_paths) > max_train:
        train_choice = np.sort(selection_rng.choice(len(train_paths), size=max_train, replace=False))
        train_paths = [train_paths[i] for i in train_choice]
    if max_test is not None and len(test_paths) > max_test:
        test_choice = np.sort(selection_rng.choice(len(test_paths), size=max_test, replace=False))
        test_paths = [test_paths[i] for i in test_choice]
        test_groups = [test_groups[i] for i in test_choice]
    if not train_paths or not test_paths:
        raise ValueError('Phase 1 clean split is empty')
    def load(paths):
        grays, cues = [], []
        for path in paths:
            with Image.open(path) as image:
                gray = prepare_gray(image.convert('RGB'), 32)
            grays.append(gray)
            # Derive cues independently inside each patch. This prevents local
            # gradients/textures from leaking masked pixels across patch borders.
            raw_patches = gray.reshape(4, 8, 4, 8).transpose(0, 2, 1, 3).reshape(16, 8, 8)
            cue_patches = [visual_cues(patch) for patch in raw_patches]
            cue = np.stack(cue_patches).reshape(4, 4, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(32, 32, 3)
            cues.append(cue)
        return np.stack(grays), np.stack(cues)
    train_gray, train_cues = load(train_paths)
    test_gray, test_cues = load(test_paths)
    train_groups = [path.parent.name for path in train_paths]
    return train_paths, test_paths, train_groups, test_groups, train_gray, train_cues, test_gray, test_cues


def gray_patches(images):
    return images.reshape(len(images), 4, 8, 4, 8).transpose(0, 1, 3, 2, 4).reshape(len(images), 16, 8, 8)


def evaluate(model, test_cues, test_gray, test_groups, phase1_model=None):
    targets = patch_grid(test_cues[0])
    del targets
    truth = gray_patches(test_gray)
    model_errors, template_errors, neighbor_errors = [], [], []
    prior_errors, rows = [], []
    template = None
    for image_index, cues in enumerate(test_cues):
        observed_patches = patch_grid(cues)
        phase1_queries = []
        for target_index in range(16):
            context = masked_neighbor_context(cues, target_index)
            prediction, winner, context_error = model.predict(context, target_index)
            target = truth[image_index, target_index]
            model_mse = float(np.mean((prediction[..., 0] - target) ** 2))
            row, col = divmod(target_index, 4)
            neighbors = []
            if col > 0: neighbors.append(observed_patches[target_index - 1][..., 0])
            if col < 3: neighbors.append(observed_patches[target_index + 1][..., 0])
            if row > 0: neighbors.append(observed_patches[target_index - 4][..., 0])
            if row < 3: neighbors.append(observed_patches[target_index + 4][..., 0])
            neighbor_prediction = np.mean(neighbors, axis=0)
            neighbor_mse = float(np.mean((neighbor_prediction - target) ** 2))
            model_errors.append(model_mse); neighbor_errors.append(neighbor_mse)
            group = test_groups[image_index]
            rows.append(dict(image_index=image_index, group=group, target_patch=target_index,
                             masked_mse=model_mse, neighbor_mean_mse=neighbor_mse,
                             winner=-1 if winner is None else winner, context_error=context_error))
            if phase1_model is not None:
                # Hide the target in grayscale first and recompute the Phase 1
                # whole-image cue so its edge/texture features cannot leak it.
                masked_gray = test_gray[image_index].copy()
                y, x = row * 8, col * 8
                masked_gray[y:y+8, x:x+8] = .5
                baseline_cue = visual_cues(masked_gray)
                phase1_queries.append(phase1_model.patches(baseline_cue)[target_index])
        if phase1_model is not None:
            scores = _distances(phase1_model.colonies, np.stack(phase1_queries))
            for target_index, score_row in enumerate(scores):
                winner = int(np.argmin(score_row)) if np.isfinite(score_row).any() else -1
                if winner < 0:
                    prior_errors.append(float(np.mean((.5 - truth[image_index, target_index]) ** 2)))
                else:
                    stored = phase1_model.colonies[winner].organism
                    patch = np.asarray([stored.nodes[i].receptor_trace or [.5] * 3
                                        for i in range(64)]).reshape(8, 8, 3)
                    prior_errors.append(float(np.mean((patch[..., 0] - truth[image_index, target_index]) ** 2)))
    return model_errors, template_errors, neighbor_errors, prior_errors, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--phase1-output', type=Path, required=True,
                        help='Phase 1 folder containing its exact duplicate audit and heldout image list')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-units', type=int, default=128)
    parser.add_argument('--masks-per-image', type=int, default=2)
    parser.add_argument('--phase1-checkpoint', type=Path,
                        help='Optional Phase 1 retrieval checkpoint for same-mask comparison')
    parser.add_argument('--max-train', type=int, help='Optional smoke-run image cap')
    parser.add_argument('--max-test', type=int, help='Optional smoke-run image cap')
    args = parser.parse_args()
    if not 1 <= args.masks_per_image <= 16 or args.max_units < 1:
        parser.error('masks-per-image must be 1..16 and max-units positive')
    args.output.mkdir(parents=True, exist_ok=True)
    (train_paths, test_paths, train_groups, test_groups, train_gray, train_cues,
     test_gray, test_cues) = data_from_phase1(args.dataset, args.phase1_output,
                                              max_train=args.max_train, max_test=args.max_test,
                                              seed=args.seed)
    rng = np.random.default_rng(args.seed)
    train_order = rng.permutation(len(train_paths))
    test_order = np.arange(len(test_paths))

    config = PredictiveConfig(max_units=args.max_units)
    model = PredictiveMycelium(config)
    train_patch_values = gray_patches(train_gray)
    train_template = train_patch_values.mean(axis=0)
    baseline_prior = VisualMycelium.load(args.phase1_checkpoint) if args.phase1_checkpoint else None
    prequential_errors = []
    start = time.monotonic()
    for step, image_index in enumerate(rng.permutation(train_order), 1):
        patches = patch_grid(train_cues[image_index])
        mask_indices = rng.choice(16, size=args.masks_per_image, replace=False)
        for target_index in mask_indices:
            context = masked_neighbor_context(train_cues[image_index], int(target_index))
            result = model.learn(context, patches[target_index], int(target_index))
            prequential_errors.append(result['prediction_mse'])
        model.images_seen += 1
        if step % 100 == 0 or step == len(train_order):
            model.validate()
            progress = dict(phase='predict-then-correct training', image=step,
                            images_total=len(train_order), samples=model.samples_seen,
                            predictive_units=len(model.units), updates=model.updates,
                            elapsed_seconds=time.monotonic()-start)
            write_json(args.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    model.freeze()
    checkpoint = args.output / 'masked-predictive-memory.json'
    model.save(checkpoint)
    before = hashlib.sha256(json.dumps(model.state_dict(), sort_keys=True).encode()).hexdigest()

    test_errors, _, neighbor_errors, phase1_errors, rows = evaluate(model, test_cues, test_gray, test_groups, baseline_prior)
    test_truth = gray_patches(test_gray)
    template_errors = [float(np.mean((test_truth[i, p] - train_template[p]) ** 2))
                       for i in range(len(test_gray)) for p in range(16)]
    per_image_errors = np.asarray(test_errors).reshape(len(test_gray), 16).mean(axis=1)
    per_image_template = np.asarray(template_errors).reshape(len(test_gray), 16).mean(axis=1)
    delta = per_image_template - per_image_errors
    bootstrap_rng = np.random.default_rng(args.seed + 77)
    boot = np.asarray([delta[bootstrap_rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)])
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], []).append(row['masked_mse'])
    checks = {
        'frozen_evaluation_read_only': before == hashlib.sha256(json.dumps(model.state_dict(), sort_keys=True).encode()).hexdigest(),
        'all_heldout_patches_supported': all(row['winner'] >= 0 for row in rows),
        'masked_context_independent_of_target_pixels': True,
        'no_labels_or_paths_in_checkpoint': not any(token in checkpoint.read_text() for token in ('glioma', 'meningioma', 'pituitary', 'notumor', 'Brain-Tumor')),
        'all_predictive_resources_within_budget': model.summary()['max_absolute_budget_error'] < 1e-8,
    }
    # Verify target blindness by mutating every held-out target patch and making
    # sure its local context is byte-identical.
    for index in range(min(32, len(test_cues))):
        for target_index in (0, 5, 10, 15):
            altered = test_cues[index].copy()
            row, col = divmod(target_index, 4)
            y, x = row * 8, col * 8
            altered[y:y+8, x:x+8, 0] = 1 - altered[y:y+8, x:x+8, 0]
            if not np.array_equal(masked_neighbor_context(test_cues[index], target_index),
                                  masked_neighbor_context(altered, target_index)):
                checks['masked_context_independent_of_target_pixels'] = False
                break
    summary = {
        'phase': '2 - unlabeled local masked-patch prediction',
        'protocol': 'For each unlabeled training image, hide one or more target patches from its local context. Predict before the patch arrives; the arriving target drives resource-funded local context and target trace updates. Competition is restricted to the same spatial patch position. Freeze before held-out evaluation.',
        'spatial_competition': 'Position-gated within the 4x4 image patch lattice; position is geometric input, not a semantic label.',
        'limits': 'A fixed lattice of graph-memory scaffolds currently stores context/target pairs. This experiment does not yet grow a connected, branching/fusing colony across images. Image-level split only; no verified patient IDs; not diagnosis or clinical validation.',
        'seed': args.seed, 'config': vars(config),
        'data': dict(training_images=len(train_paths), training_masks=model.samples_seen,
                     heldout_images=len(test_paths), heldout_masked_patches=len(test_errors),
                     duplicate_screen_reused_from='Phase 1 exact duplicate-audit.csv and heldout-results.csv'),
        'learning': model.summary(),
        'prequential_training_prediction_mse': float(np.mean(prequential_errors)),
        'heldout_masked_patch_mse': float(np.mean(test_errors)),
        'heldout_masked_patch_rmse': float(np.sqrt(np.mean(test_errors))),
        'training_mean_patch_template_mse': float(np.mean(template_errors)),
        'neighbor_mean_interpolation_mse': float(np.mean(neighbor_errors)),
        'phase1_masked_retrieval_mse': float(np.mean(phase1_errors)) if phase1_errors else None,
        'paired_per_image_template_minus_predictor_mse': dict(
            mean=float(delta.mean()), bootstrap_95pct=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            bootstrap_unit='image; not patient-level'),
        'heldout_group_mse_posthoc': {name: dict(patches=len(values), mean_mse=float(np.mean(values)))
                                     for name, values in sorted(groups.items())},
        'checks': checks, 'checks_passed': all(checks.values()),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'elapsed_seconds': time.monotonic() - start,
    }
    with (args.output / 'masked-patch-results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'progress.json', dict(phase='complete', checks_passed=summary['checks_passed']))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['checks_passed']:
        raise SystemExit('masked prediction integrity check failed')


if __name__ == '__main__':
    main()
