"""Full-scale run of the best Training-probe configuration (64px edge/texture).

Unlabeled MYCELIA fit on all clean Training images, spatial-pyramid colony
features, then the same Training-only RBF-SVM protocol as the HOG baseline and
tools/matched_readout_comparison.py. Testing labels are read only after the
model, features, and C are fixed. Retrospective: Testing was scored before.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))

import numpy as np
from mycelia.omnia_image import load_omnia_rgb
from mycelia.visual_memory import VisualConfig, VisualMycelium, prepare_gray_fixed, visual_cues
from mycelia.visual_readout import spatial_pyramid_features
from omnia_tumor_research import (clean_image_splits, load_omnia_dataset, read_test_labels,
                                  stratified_validation, write_json)
from matched_readout_comparison import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--omnia-sdk', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--capacity', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records, _, _ = load_omnia_dataset(a.dataset_root, a.omnia_sdk)
    train, test, audit = clean_image_splits(records)
    inner, validation = stratified_validation(records, train, a.seed)
    used = np.concatenate([train, test])
    cues = np.zeros((len(records), a.size, a.size, 3))
    for n, i in enumerate(used, 1):
        image = load_omnia_rgb(Path(a.dataset_root)/'dataset'/records[int(i)]['file'])
        cues[int(i)] = visual_cues(prepare_gray_fixed(image, a.size), detailed=True,
                                   multiscale=True, mode='edge_texture')
        if n % 1000 == 0:
            print(f'encoded {n}/{len(used)}', flush=True)
    config = VisualConfig(size=a.size, patch_size=8, max_colonies=a.capacity,
                          max_episodes=len(train), preprocessing='fixed',
                          multiscale_features=True, cue_mode='edge_texture',
                          hierarchical_context=False, specialization=True, error_gated=True)
    model = VisualMycelium(config)
    fit_start = time.perf_counter()
    for n, i in enumerate(np.random.default_rng(a.seed).permutation(train), 1):
        model.learn(cues[int(i)])
        if n % 250 == 0:
            write_json(a.output/'progress.json', {'images_learned': n, 'of': len(train),
                       'colonies': model.colony_count,
                       'elapsed_s': time.perf_counter()-fit_start})
            print(f'learned {n}/{len(train)} colonies={model.colony_count}', flush=True)
    model.freeze(); model.validate()
    fit_seconds = time.perf_counter()-fit_start
    model.save(a.output/'mycelia-64px-memory.json')
    levels = (1, 2, 4) if a.size == 32 else (1, 2, 4, 8)
    feats = spatial_pyramid_features(model, cues[used], levels=levels)
    x = np.zeros((len(records), feats.shape[1])); x[used] = feats
    actual = read_test_labels(a.dataset_root, records, test)
    result, _ = evaluate(f'mycelia_{a.size}px_svm', x, records, inner, validation,
                         train, test, actual)
    summary = {
        'schema': 'mycelia.omnia-64px-full-run.v1', 'scope': __doc__.strip(),
        'size': a.size, 'capacity': a.capacity, 'pyramid_levels': list(levels),
        'counts': {'train': len(train), 'test': len(test)}, 'deduplication': audit,
        'unlabeled_fit_seconds': fit_seconds,
        'learning': {k: v for k, v in model.summary().items()
                     if k not in ('current_colony_support', 'selections')},
        'labels_passed_to_visual_learning': False,
        'svm_readout': result, 'elapsed_seconds': time.perf_counter()-started,
    }
    write_json(a.output/'summary.json', summary)
    print(json.dumps(result['heldout_metrics'], indent=1)[:600])


if __name__ == '__main__':
    main()
