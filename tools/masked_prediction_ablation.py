"""Evaluate whether masked-patch memory uses context, position and learned cords."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mycelia.predictive_memory import PredictiveMycelium, masked_neighbor_context, patch_grid
from mycelia.visual_memory import prepare_gray, visual_cues


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def load_test(dataset, phase1_output):
    with (phase1_output / 'heldout-results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    gray, cues = [], []
    for row in rows:
        with Image.open(dataset / row['file']) as image:
            image_gray = prepare_gray(image.convert('RGB'), 32)
        gray.append(image_gray)
        patches = image_gray.reshape(4, 8, 4, 8).transpose(0, 2, 1, 3).reshape(16, 8, 8)
        isolated = np.stack([visual_cues(patch) for patch in patches])
        cues.append(isolated.reshape(4, 4, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(32, 32, 3))
    truth = np.stack(gray).reshape(len(gray), 4, 8, 4, 8).transpose(0, 1, 3, 2, 4).reshape(len(gray), 16, 8, 8)
    return rows, np.stack(cues), truth


def digest(model):
    return hashlib.sha256(json.dumps(model.state_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()


def score(model, cues, truth, context_sources=None, position_offset=0):
    errors = []
    n = len(cues)
    for i, cue in enumerate(cues):
        patches = patch_grid(cue)
        source = i if context_sources is None else int(context_sources[i])
        source_cue = cue if source == i else cues[source]
        for target in range(16):
            context = masked_neighbor_context(source_cue, target)
            position = (target + position_offset) % 16
            predicted, winner, _ = model.predict(context, position)
            if winner is None:
                predicted_intensity = np.full((8, 8), .5)
            else:
                predicted_intensity = predicted[..., 0]
            errors.append(float(np.mean((predicted_intensity - truth[i, target]) ** 2)))
    return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--phase1-output', type=Path, required=True)
    parser.add_argument('--phase2-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, cues, truth = load_test(args.dataset, args.phase1_output)
    model = PredictiveMycelium.load(args.phase2_output / 'masked-predictive-memory.json')
    if not model.frozen:
        raise ValueError('ablation requires a frozen memory')
    initial = digest(model)
    actual = score(model, cues, truth)
    rng = np.random.default_rng(20261004)
    order = rng.permutation(len(cues))
    sources = np.empty(len(cues), dtype=int)
    sources[order] = np.roll(order, 1)
    shuffled_context = score(model, cues, truth, context_sources=sources)
    wrong_position = score(model, cues, truth, position_offset=1)

    saved = []
    for unit in model.units:
        for edge in unit.context.organism.segments.values():
            saved.append((edge, edge.route_trace, edge.radius))
            edge.route_trace = 0.0
            edge.radius = unit.context.organism.config.radius * .7
    model._cache_dirty = True
    cord_disabled = score(model, cues, truth)
    for edge, route, radius in saved:
        edge.route_trace, edge.radius = route, radius
    model._cache_dirty = True
    restored = digest(model)
    result = {
        'heldout_images': len(rows), 'masked_patches': len(rows) * 16,
        'same_image_context_mse': actual,
        'context_permuted_across_heldout_images_mse': shuffled_context,
        'context_permutation_mse_increase': shuffled_context - actual,
        'position_shifted_by_one_patch_mse': wrong_position,
        'position_shift_mse_increase': wrong_position - actual,
        'learned_context_cords_disabled_mse': cord_disabled,
        'cord_ablation_mse_increase': cord_disabled - actual,
        'cord_ablation_changed_prediction_quality': abs(cord_disabled - actual) > 1e-12,
        'ablation_restored_frozen_model_exactly': initial == restored,
        'context_permutation': 'Held-out contexts were deranged across images separately at each patch position; targets remained paired with their original images.',
        'position_ablation': 'Each visible context was queried at the next spatial patch location.',
        'cord_ablation': 'Set learned context-cord trace to zero and cord radius to its unreinforced baseline, then restored the checkpoint state.',
        'limit': 'Image-level, not patient-level; the ablations test this computational model, not fungal biological equivalence.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'context-ablation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
