"""Recall stored local visual patterns from an image cue; no category prediction."""
import argparse
import json
from pathlib import Path
import sys

# Support direct ``python tools/...`` execution from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from mycelia.visual_memory import VisualMycelium


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = VisualMycelium.load(args.model)
    with Image.open(args.image) as image:
        cues = model.encode_image(image)
        gray = cues[..., 0]
    identity = model.recognize(cues)
    recalled, assignments = model.reconstruct(cues)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(recalled[..., 0], 0, 1)*255).astype(np.uint8)).save(args.output)
    print(json.dumps(dict(scope='Cue-addressed visual recall; no tumor categories predicted.',
                          visual_identity=identity,
                          gray_rmse=float(np.sqrt(np.mean((gray-recalled[..., 0])**2))),
                          supported_patches=int(np.count_nonzero(assignments >= 0)),
                          patches=len(assignments), output=str(args.output.resolve())), indent=2))


if __name__ == '__main__': main()
