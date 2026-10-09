"""Decode the one-image-per-container OMNIA layout used by this experiment."""
from __future__ import annotations
from pathlib import Path
import io

from PIL import Image
import numpy as np


def load_omnia_rgb(path):
    try:
        from omnia_sdk.container import OmniaContainer
    except ImportError as exc:
        raise RuntimeError('OMNIA input requires omnia-sdk; install it or add its checkout to PYTHONPATH') from exc
    with OmniaContainer(Path(path)) as container:
        if container.num_slices != 1:
            raise ValueError(f'{path} must contain exactly one full-image tile')
        tile = container.get_slice(0)
        if container.pixel_codec == 'jpeg':
            image = Image.open(io.BytesIO(tile.tobytes())).convert('RGB')
            image.load()
            return image
        if container.pixel_codec == 'zstd':
            return Image.fromarray(np.asarray(tile)).convert('RGB')
        raise ValueError(f'unsupported OMNIA pixel codec: {container.pixel_codec!r}')
