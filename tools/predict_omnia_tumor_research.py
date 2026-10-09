"""Predict a research dataset label for one full-resolution OMNIA MRI image."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from mycelia.visual_memory import VisualMycelium
from mycelia.omnia_image import load_omnia_rgb
from mycelia.visual_readout import colony_match_features, centroid_distance_scores


def predict_file(image_path, model_dir, sdk_path=None, compiled_model=None):
    image_path = Path(image_path)
    model_dir = Path(model_dir)
    if sdk_path:
        sys.path.insert(0, str(Path(sdk_path).resolve()))
    memory_path = model_dir/'unsupervised-mycelia-memory.json'
    readout_path = model_dir/'external-nearest-centroid-readout.json'
    readout = json.loads(readout_path.read_text())
    if readout.get('schema') != 'mycelia.omnia-colony-centroid.v1':
        raise ValueError('unsupported readout schema')
    if compiled_model is None:
        actual_hash = hashlib.sha256(memory_path.read_bytes()).hexdigest()
        if readout.get('memory_sha256') != actual_hash:
            raise ValueError('readout and MYCELIA memory checkpoints do not match')
        model = VisualMycelium.load(memory_path)
    else:
        from mycelia.frozen_visual import FrozenVisualMatcher
        model = FrozenVisualMatcher.load(compiled_model)
        if readout.get('memory_sha256') != model.source_memory_sha256:
            raise ValueError('readout and compiled MYCELIA checkpoints do not match')
    if not model.frozen:
        raise ValueError('MYCELIA checkpoint must be frozen before prediction')
    image = load_omnia_rgb(image_path)
    cue = model.encode_image(image)
    features = colony_match_features(model, cue[None, ...])
    if features.shape[1] != readout.get('feature_dimension'):
        raise ValueError('input feature dimension does not match the saved readout')
    scores = centroid_distance_scores(readout, features)[0]
    order = np.argsort(scores, kind='stable')
    labels = readout['labels']
    result = {
        'image': str(image_path),
        'predicted_dataset_label': labels[int(order[0])],
        'readout': 'Training-only standardized nearest centroid over frozen label-free MYCELIA features',
        'class_distance_scores': {label: float(scores[i]) for i, label in enumerate(labels)},
        'nearest_centroid_distance_margin': float(scores[order[1]]-scores[order[0]]),
        'score_interpretation': 'Standardized squared distances; not probabilities or calibrated clinical confidence.',
        'scope': 'Research-only four-class image-level dataset prediction; not a clinical diagnosis or tumor localization.',
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, help='one full-resolution .omnia MRI container')
    parser.add_argument('--model-dir', type=Path, required=True,
                        help='training output directory containing the frozen model and readout')
    parser.add_argument('--omnia-sdk', type=Path, required=True,
                        help='path to the omnia-sdk source checkout')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compiled-model', type=Path,
                        help='optional compact frozen graph snapshot exported from this memory')
    args = parser.parse_args()
    result = predict_file(args.image, args.model_dir, args.omnia_sdk, args.compiled_model)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
