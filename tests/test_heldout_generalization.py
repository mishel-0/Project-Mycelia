import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def test_label_free_heldout_protocol_excludes_cross_split_copy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path / 'dataset'
    rng = np.random.default_rng(981)
    train_a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    for split, group, images in [
        ('Training', 'glioma', [train_a, rng.integers(0, 256, (32, 32), dtype=np.uint8),
                               rng.integers(0, 256, (32, 32), dtype=np.uint8)]),
        ('Testing', 'meningioma', [train_a, rng.integers(0, 256, (32, 32), dtype=np.uint8),
                                  rng.integers(0, 256, (32, 32), dtype=np.uint8)]),
    ]:
        folder = dataset / split / group
        folder.mkdir(parents=True)
        for index, image in enumerate(images):
            Image.fromarray(image).save(folder / f'{index}.png')

    output = tmp_path / 'results'
    subprocess.run([
        sys.executable, str(root / 'tools/heldout_generalization_test.py'),
        '--dataset', str(dataset), '--output', str(output), '--max-colonies', '4',
    ], cwd=root, check=True, capture_output=True, text=True)
    summary = json.loads((output / 'summary.json').read_text())
    checkpoint = json.loads((output / 'phase1-unlabeled-memory.json').read_text())

    assert summary['checks_passed']
    assert summary['data']['training_used'] == 3
    assert summary['data']['heldout_test_evaluated'] == 2
    assert summary['data']['test_exclusion_reasons']['cross_split_duplicate_source_pixels'] == 1
    assert summary['checks']['frozen_memory_unchanged_by_evaluation']
    assert 'glioma' not in json.dumps(checkpoint)
    assert 'meningioma' not in json.dumps(checkpoint)
