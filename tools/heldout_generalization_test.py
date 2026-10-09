"""Train MYCELIA on unlabeled MRI Training images and test on unseen Testing images.

This is an image-level generalization experiment. Folder names are retained only
for the final external audit and never passed to VisualMycelium.learn(). It is
not patient-independent validation or a medical diagnostic test.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from mycelia.visual_memory import VisualConfig, VisualMycelium, prepare_gray, visual_cues


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
NEAR_DUPLICATE_BITS = 4


def write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def _perceptual_hash(gray: np.ndarray) -> int:
    """Small DCT perceptual hash, independent of class or file metadata."""
    small = np.asarray(Image.fromarray(np.rint(gray * 255).astype(np.uint8)).resize(
        (32, 32), Image.Resampling.BILINEAR), dtype=float)
    n = small.shape[0]
    axis = np.arange(n)
    basis = np.cos(np.pi * (2 * axis[None, :] + 1) * axis[:, None] / (2 * n))
    basis[0] *= 1 / np.sqrt(2)
    basis *= np.sqrt(2 / n)
    low = (basis @ small @ basis.T)[:8, :8].ravel()[1:]
    median = float(np.median(low))
    value = 0
    for bit, coefficient in enumerate(low):
        if coefficient > median:
            value |= 1 << bit
    return value


def _near_any(value: int, references: list[int], max_bits: int = NEAR_DUPLICATE_BITS):
    best = 65
    best_index = None
    for index, reference in enumerate(references):
        distance = (value ^ reference).bit_count()
        if distance < best:
            best = distance
            best_index = index
            if best == 0:
                return best_index, best
    return (best_index, best) if best <= max_bits else (None, best)


def load_split(dataset: Path, split: str):
    root = dataset / split
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f'no images found under {root}')
    records, gray = [], []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            source_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            prepared = prepare_gray(rgb, 32)
        records.append({
            'split': split,
            'file': str(path.relative_to(dataset)),
            # Used exclusively by the post-fit evaluator; never enters the model.
            'audit_group': path.parent.name,
            'source_sha256': source_hash,
            'cue_sha256': hashlib.sha256(prepared.tobytes()).hexdigest(),
            'perceptual_hash': _perceptual_hash(prepared),
        })
        gray.append(prepared)
    return records, np.stack(gray)


def deduplicate(records, gray, split):
    """Deterministically remove exact, encoded, and very close perceptual copies."""
    kept, excluded, sources, cues, hashes = [], [], set(), set(), []
    for index, record in enumerate(records):
        duplicate_reason = None
        if record['source_sha256'] in sources:
            duplicate_reason = 'duplicate_source_pixels'
        elif record['cue_sha256'] in cues:
            duplicate_reason = 'duplicate_32px_input'
        else:
            match, distance = _near_any(record['perceptual_hash'], hashes)
            if match is not None:
                duplicate_reason = f'near_duplicate_phash_{distance}'
        if duplicate_reason:
            excluded.append((index, duplicate_reason))
            continue
        kept.append(index)
        sources.add(record['source_sha256'])
        cues.add(record['cue_sha256'])
        hashes.append(record['perceptual_hash'])
    return np.asarray(kept, dtype=int), excluded


def fit(cues, indices, config, seed, output):
    model = VisualMycelium(config)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    for step, index in enumerate(rng.permutation(indices), 1):
        # The learner receives only a numerical cue array, never paths or labels.
        model.learn(cues[index])
        if step % 100 == 0 or step == len(indices):
            model.validate()
            progress = {
                'phase': 'unlabeled_training', 'image': step,
                'images_total': len(indices), 'colonies': len(model.colonies),
                'updates': model.updates, 'suppressed': model.suppressed_updates,
                'elapsed_seconds': time.monotonic() - started,
            }
            write_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    model.freeze()
    return model


def state_hash(model):
    payload = json.dumps(model.state_dict(), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def reconstruction(model, cue):
    result, assignments = model.reconstruct(cue)
    return result[..., 0], assignments


def evaluate(model, cues, gray, indices):
    errors, support = [], []
    for index in indices:
        recalled, assignments = reconstruction(model, cues[index])
        errors.append(float(np.mean((recalled - gray[index]) ** 2)))
        support.append(not np.any(assignments < 0))
    errors = np.asarray(errors)
    blank = float(np.mean((gray[indices] - .5) ** 2))
    mse = float(errors.mean())
    return {
        'images': int(len(indices)),
        'mean_grayscale_mse': mse,
        'grayscale_rmse': float(np.sqrt(mse)),
        'blank_gray_baseline_mse': blank,
        'mse_reduction_vs_blank': float(1 - mse / max(blank, 1e-12)),
        'fully_supported_images': int(sum(support)),
    }, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-colonies', type=int, default=256)
    parser.add_argument('--correction-threshold', type=float, default=.004)
    parser.add_argument('--multiscale-features', action=argparse.BooleanOptionalAction, default=True,
                        help='use broad-range edge/texture cues in addition to intensity')
    parser.add_argument('--max-train', type=int, help='Optional smaller smoke run; leave unset for all clean training images.')
    parser.add_argument('--max-test', type=int, help='Optional smaller smoke run; leave unset for every clean held-out image.')
    args = parser.parse_args()
    if args.max_colonies < 1 or (args.max_train is not None and args.max_train < 1) or (args.max_test is not None and args.max_test < 1):
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    train_records, train_gray = load_split(args.dataset, 'Training')
    test_records, test_gray = load_split(args.dataset, 'Testing')
    train_keep, train_excluded = deduplicate(train_records, train_gray, 'Training')
    train_hashes = [train_records[i]['perceptual_hash'] for i in train_keep]
    train_sources = {train_records[i]['source_sha256'] for i in train_keep}
    train_cues = {train_records[i]['cue_sha256'] for i in train_keep}

    # Remove any held-out image matching training content, then deduplicate the
    # remaining test set without consulting its category names.
    cross_excluded, remaining = [], []
    for index, record in enumerate(test_records):
        if record['source_sha256'] in train_sources:
            cross_excluded.append((index, 'cross_split_duplicate_source_pixels'))
        elif record['cue_sha256'] in train_cues:
            cross_excluded.append((index, 'cross_split_duplicate_32px_input'))
        else:
            match, distance = _near_any(record['perceptual_hash'], train_hashes)
            if match is not None:
                cross_excluded.append((index, f'cross_split_near_duplicate_phash_{distance}'))
            else:
                remaining.append(index)
    test_subset_records = [test_records[i] for i in remaining]
    test_subset_gray = test_gray[remaining]
    test_keep_local, test_excluded_local = deduplicate(test_subset_records, test_subset_gray, 'Testing')
    test_keep = np.asarray(remaining, dtype=int)[test_keep_local]
    test_excluded = cross_excluded + [(remaining[i], reason) for i, reason in test_excluded_local]

    # Persist the split audit before the long training pass, so the leakage
    # filtering itself is reviewable even if a machine interruption occurs.
    with (args.output / 'duplicate-audit.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'file', 'reason'])
        for index, reason in train_excluded:
            writer.writerow(['Training', train_records[index]['file'], reason])
        for index, reason in test_excluded:
            writer.writerow(['Testing', test_records[index]['file'], reason])
    audit = {
        'phase': 'duplicate audit complete',
        'training_raw': len(train_records), 'training_clean': len(train_keep),
        'testing_raw': len(test_records), 'testing_clean': len(test_keep),
        'training_excluded_reasons': dict(__import__('collections').Counter(reason for _, reason in train_excluded)),
        'testing_excluded_reasons': dict(__import__('collections').Counter(reason for _, reason in test_excluded)),
    }
    write_json(args.output / 'progress.json', audit)
    print(json.dumps(audit), flush=True)

    rng = np.random.default_rng(args.seed)
    train_order = rng.permutation(train_keep)
    if args.max_train is not None:
        train_order = train_order[:args.max_train]
    test_order = np.random.default_rng(args.seed + 1).permutation(test_keep)
    if args.max_test is not None:
        test_order = test_order[:args.max_test]

    train_cues_array = np.stack([visual_cues(image, multiscale=args.multiscale_features) for image in train_gray])
    test_cues_array = np.stack([visual_cues(image, multiscale=args.multiscale_features) for image in test_gray])
    config = VisualConfig(max_colonies=args.max_colonies,
                          correction_threshold=args.correction_threshold,
                          multiscale_features=args.multiscale_features)
    model = fit(train_cues_array, train_order, config, args.seed, args.output)
    checkpoint = args.output / 'phase1-unlabeled-memory.json'
    model.save(checkpoint)
    before = state_hash(model)

    # The score is reported for all clean held-out images unless a smoke cap was
    # explicitly supplied. Folder labels are read only now, by the audit code.
    test_result, test_errors = evaluate(model, test_cues_array, test_gray, test_order)
    seen_sample = np.random.default_rng(args.seed + 2).choice(train_order, min(256, len(train_order)), replace=False)
    seen_result, _ = evaluate(model, train_cues_array, train_gray, seen_sample)
    training_template = train_gray[train_order].mean(axis=0)
    template_mse = float(np.mean((test_gray[test_order] - training_template) ** 2))
    training_mean = float(train_gray[train_order].mean())
    constant_training_mean_mse = float(np.mean((test_gray[test_order] - training_mean) ** 2))
    group_rows = {}
    for index, error in zip(test_order, test_errors):
        group = test_records[index]['audit_group']
        group_rows.setdefault(group, []).append(float(error))
    checks = {
        'clean_train_test_source_hashes_disjoint': not (train_sources & {test_records[i]['source_sha256'] for i in test_keep}),
        'clean_train_test_32px_cues_disjoint': not (train_cues & {test_records[i]['cue_sha256'] for i in test_keep}),
        'heldout_is_unseen_to_learner': model.images_seen == len(train_order),
        'frozen_memory_unchanged_by_evaluation': state_hash(model) == before,
        'all_test_reconstructions_supported': test_result['fully_supported_images'] == len(test_order),
    }
    summary = {
        'phase': '1 - unlabeled held-out generalization',
        'scope': 'Train on Training images without labels; freeze; reconstruct clean Testing images never used for learning.',
        'limit': 'Image-level split only. Dataset has no verified patient identifiers, so patient-independent medical generalization is not established. Reconstruction is not tumor classification or clinical validation.',
        'seed': args.seed,
        'config': asdict(config),
        'data': {
            'training_raw': len(train_records), 'training_clean': len(train_keep),
            'training_duplicates_excluded': len(train_excluded),
            'testing_raw': len(test_records), 'testing_clean': len(test_keep),
            'testing_duplicates_excluded': len(test_excluded),
            'training_used': len(train_order), 'heldout_test_evaluated': len(test_order),
            'near_duplicate_threshold_phash_bits': NEAR_DUPLICATE_BITS,
            'training_exclusion_reasons': dict(__import__('collections').Counter(reason for _, reason in train_excluded)),
            'test_exclusion_reasons': dict(__import__('collections').Counter(reason for _, reason in test_excluded)),
        },
        'learning': model.summary(),
        'seen_training_sample': seen_result,
        'heldout_test': test_result,
        'additional_unlabeled_baselines': {
            'training_mean_image_template_mse': template_mse,
            'training_mean_image_template_relative_reduction': float(1 - test_result['mean_grayscale_mse'] / max(template_mse, 1e-12)),
            'training_global_mean_constant_mse': constant_training_mean_mse,
        },
        'heldout_audit_groups': {
            group: {'images': len(errors), 'mean_grayscale_mse': float(np.mean(errors))}
            for group, errors in sorted(group_rows.items())
        },
        'checks': checks,
        'checks_passed': all(checks.values()),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'elapsed_seconds': time.monotonic() - started,
    }
    with (args.output / 'heldout-results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file', 'audit_group', 'gray_mse', 'excluded_from_training'])
        for index, error in zip(test_order, test_errors):
            writer.writerow([test_records[index]['file'], test_records[index]['audit_group'], error, 0])
    with (args.output / 'duplicate-audit.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'file', 'reason'])
        for index, reason in train_excluded:
            writer.writerow(['Training', train_records[index]['file'], reason])
        for index, reason in test_excluded:
            writer.writerow(['Testing', test_records[index]['file'], reason])
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'progress.json', {'phase': 'complete', 'checks_passed': summary['checks_passed']})
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['checks_passed']:
        raise SystemExit('Phase 1 integrity check failed')


if __name__ == '__main__':
    main()
