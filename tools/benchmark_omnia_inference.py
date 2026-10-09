"""Verify compiled MYCELIA inference and measure its cost on this machine.

This replays saved predictions to check an execution optimization; it does not
select, retrain, or claim a new diagnostic model. No accuracy target is tuned.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from mycelia.frozen_visual import FrozenVisualMatcher
from mycelia.omnia_image import load_omnia_rgb
from mycelia.visual_memory import VisualMycelium
from mycelia.visual_readout import colony_match_features, centroid_distance_scores


def timed(call, repeats=5):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        times.append(time.perf_counter()-started)
    return {'median_seconds': float(np.median(times)), 'samples_seconds': times}


def benchmark(model_dir, compiled_path, dataset_root, output, sdk_path):
    sys.path.insert(0, str(Path(sdk_path).resolve()))
    model_dir, dataset_root, output = map(Path, (model_dir, dataset_root, output))
    output.parent.mkdir(parents=True, exist_ok=True)
    memory = model_dir/'unsupervised-mycelia-memory.json'
    readout = json.loads((model_dir/'external-nearest-centroid-readout.json').read_text())
    digest = hashlib.sha256(memory.read_bytes()).hexdigest()
    started = time.perf_counter()
    model = VisualMycelium.load(memory)
    full_load = time.perf_counter()-started
    started = time.perf_counter()
    compiled = FrozenVisualMatcher.load(compiled_path)
    compact_load = time.perf_counter()-started
    if digest != readout['memory_sha256'] or digest != compiled.source_memory_sha256:
        raise ValueError('benchmark artifacts do not belong to the same frozen model')
    rows = list(csv.DictReader((model_dir/'heldout-predictions.csv').open(newline='')))
    # Full saved-cohort replay verifies labels and class scores without making
    # training or hyperparameter choices from these known held-out outcomes.
    same_predictions = 0
    max_score_difference = 0.0
    feature_identity = True
    decode_seconds = 0.0
    for start in range(0, len(rows), 64):
        part = rows[start:start+64]
        began = time.perf_counter()
        cues = np.stack([compiled.encode_image(load_omnia_rgb(dataset_root/'dataset'/row['file']))
                         for row in part])
        decode_seconds += time.perf_counter()-began
        features = colony_match_features(compiled, cues)
        reference = colony_match_features(model, cues)
        feature_identity &= np.array_equal(features, reference)
        scores = centroid_distance_scores(readout, features)
        predicted = np.asarray(readout['labels'])[scores.argmin(axis=1)]
        expected = np.asarray([[float(row['distance_'+label]) for label in readout['labels']]
                               for row in part])
        max_score_difference = max(max_score_difference, float(np.max(np.abs(scores-expected))))
        same_predictions += sum(str(predicted[i]) == row['prediction'] for i, row in enumerate(part))
        if start == 0:
            sample = cues
    if not feature_identity or same_predictions != len(rows) or max_score_difference > 1e-8:
        raise AssertionError('compact model does not preserve saved inference')
    timings = []
    for count in (1, 8, 32, 64):
        batch = sample[:count]
        for backend in (model, compiled):
            colony_match_features(backend, batch)
        full = timed(lambda: colony_match_features(model, batch))
        compact = timed(lambda: colony_match_features(compiled, batch))
        timings.append({'images': len(batch), 'full_graph': full, 'compiled_graph': compact,
                        'median_speedup': full['median_seconds']/compact['median_seconds']})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == 'darwin' else peak*1024
    result = {
        'scope': 'Replay of an already evaluated frozen model; inference equivalence and performance only.',
        'platform': platform.platform(), 'python': platform.python_version(),
        'numpy': np.__version__, 'source_memory_sha256': digest,
        'checkpoint_bytes': {'full': memory.stat().st_size, 'compiled': Path(compiled_path).stat().st_size},
        'load_seconds_single_measurement': {'full': full_load, 'compiled': compact_load},
        'numeric_array_bytes': compiled.array_bytes,
        'parity': {'images': len(rows), 'same_predictions': same_predictions,
                   'features_identical': bool(feature_identity),
                   'maximum_saved_class_score_difference': max_score_difference},
        'decode_and_encode_seconds_for_replay': decode_seconds,
        'warm_feature_timings': timings,
        'process_peak_rss_bytes': int(peak_bytes),
        'memory_scope': 'Whole benchmark process with both models resident, SDK and working arrays; not isolated inference/training peak.',
        'limits': 'Warm timings exclude image decoding and model loading. Compilation changes neither training cost nor diagnostic accuracy. Hardware-specific, not a claim of superiority to other AI models.',
    }
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--compiled-model', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--omnia-sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.model_dir, args.compiled_model, args.dataset_root,
                       args.output, args.omnia_sdk)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
