"""Measure the full-resolution information ceiling of MYCELIA's 32px input."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from mycelia.omnia_image import load_omnia_rgb
from mycelia.visual_memory import VisualMycelium


def invert_fixed_canvas(gray32, original_size):
    width, height = original_size
    ratio = min(32/height, 32/width)
    fitted_w, fitted_h = max(1, round(width*ratio)), max(1, round(height*ratio))
    y, x = (32-fitted_h)//2, (32-fitted_w)//2
    crop = np.asarray(gray32[y:y+fitted_h, x:x+fitted_w], dtype=np.float32)
    return np.asarray(Image.fromarray(crop, mode='F').resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def image_error(reference, reconstruction):
    delta = reference-reconstruction
    mse = float(np.mean(delta*delta))
    return {'mse': mse, 'rmse': math.sqrt(mse),
            'psnr_db': float('inf') if mse == 0 else -10*math.log10(mse),
            'exact_pixel_fraction': float(np.mean(reference == reconstruction))}


def run(dataset_root, sdk_path, memory_path, output, sample_count=256, seed=4211):
    sys.path.insert(0, str(Path(sdk_path).resolve()))
    dataset_root, memory_path = Path(dataset_root), Path(memory_path)
    rows = list(csv.DictReader((dataset_root/'manifest.csv').open(newline='')))
    test = [row for row in rows if row['split'] == 'Testing']
    chosen = np.random.default_rng(seed).choice(len(test), min(sample_count, len(test)), replace=False)
    model = VisualMycelium.load(memory_path)
    upper_errors, learned_errors, reductions = [], [], []
    started = time.perf_counter()
    for index in chosen:
        image = load_omnia_rgb(dataset_root/test[int(index)]['omnia'])
        reference = np.asarray(image.convert('L'), dtype=np.float32)/255
        cue = model.encode_image(image)
        upper = invert_fixed_canvas(cue[..., 0], image.size)
        recalled, _ = model.reconstruct(cue)
        learned = invert_fixed_canvas(recalled[..., 0], image.size)
        upper_errors.append(image_error(reference, upper))
        learned_errors.append(image_error(reference, learned))
        reductions.append(reference.size/(32*32))
    def aggregate(items):
        return {key: float(np.mean([row[key] for row in items])) for key in items[0]}
    result = {
        'schema': 'mycelia.full-resolution-information-audit.v1',
        'sampled_testing_images': len(chosen), 'seed': seed,
        'original_pixels_per_32px_cue_mean_ratio': float(np.mean(reductions)),
        'preprocessing_upper_bound': aggregate(upper_errors),
        'learned_colony_reconstruction': aggregate(learned_errors),
        'source_exact_reconstruction_supported_by_current_model': False,
        'reason': 'The learner receives a 32x32x3 derived cue and cannot infer pixels discarded by resizing. The preprocessing upper bound already differs from source pixels.',
        'omnia_role': 'OMNIA preserves the source image payload for decoding; that is storage fidelity, not learned reconstruction.',
        'elapsed_seconds': time.perf_counter()-started,
    }
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--omnia-sdk', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=256)
    args = parser.parse_args()
    if args.sample_count < 1: parser.error('--sample-count must be positive')
    print(json.dumps(run(args.dataset_root, args.omnia_sdk, args.memory,
                         args.output, args.sample_count), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
