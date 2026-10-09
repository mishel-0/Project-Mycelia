import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from mycelia.predictive_memory import PredictiveConfig, PredictiveMycelium, masked_neighbor_context


def test_masked_context_cannot_read_the_hidden_patch():
    cue = np.random.default_rng(41).random((32, 32, 3))
    altered = cue.copy()
    altered[8:16, 8:16] = 1 - altered[8:16, 8:16]
    first = masked_neighbor_context(cue, 5)
    second = masked_neighbor_context(altered, 5)
    assert np.array_equal(first, second)


def test_prediction_precedes_local_error_update_and_checkpoint_is_read_only(tmp_path):
    model = PredictiveMycelium(PredictiveConfig(size=8, patch_size=8, max_units=2))
    context = np.full((8, 8, 3), .3)
    target = np.full((8, 8, 3), .9)
    initial, winner, _ = model.predict(context)
    assert winner is None and np.all(initial == .5)
    result = model.learn(context, target)
    assert result['prediction_mse'] == pytest.approx(.16)
    prediction, winner, _ = model.predict(context)
    assert winner == 0
    assert np.mean((prediction[..., 0] - .9) ** 2) < result['prediction_mse']
    before = model.state_dict()
    model.freeze()
    before_frozen = model.state_dict()
    model.predict(context)
    assert model.state_dict() == before_frozen
    checkpoint = tmp_path / 'predictive.json'
    model.save(checkpoint)
    restored = PredictiveMycelium.load(checkpoint)
    assert restored.predict(context)[0].tolist() == model.predict(context)[0].tolist()
    assert restored.state_dict() == model.state_dict()
    with pytest.raises(RuntimeError):
        model.learn(context, target)
    assert before['samples_seen'] == 1
    assert model.summary()['max_absolute_budget_error'] < 1e-8


def test_spatial_gating_keeps_same_context_targets_at_distinct_locations_separate():
    model = PredictiveMycelium(PredictiveConfig(size=16, patch_size=8, max_units=4))
    context = np.full((8, 8, 3), .4)
    bright = np.full((8, 8, 3), .9)
    dark = np.full((8, 8, 3), .1)
    model.learn(context, bright, position=0)
    model.learn(context, dark, position=1)
    first, first_unit, _ = model.predict(context, position=0)
    second, second_unit, _ = model.predict(context, position=1)
    assert first_unit != second_unit
    assert np.mean((first[..., 0] - .9) ** 2) < .01
    assert np.mean((second[..., 0] - .1) ** 2) < .01


def test_masked_prediction_cli_keeps_paths_out_of_learned_state(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path / 'dataset'
    rng = np.random.default_rng(101)
    for split, name, count in [('Training', 'glioma', 2), ('Testing', 'pituitary', 1)]:
        folder = dataset / split / name
        folder.mkdir(parents=True)
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8)).save(folder / f'{index}.png')
    phase1 = tmp_path / 'phase1'
    phase1.mkdir()
    with (phase1 / 'duplicate-audit.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['split', 'file', 'reason'])
    with (phase1 / 'heldout-results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['file', 'audit_group', 'gray_mse'])
        writer.writerow(['Testing/pituitary/0.png', 'pituitary', .1])
    output = tmp_path / 'result'
    subprocess.run([
        sys.executable, str(root / 'tools/masked_patch_prediction_test.py'),
        '--dataset', str(dataset), '--phase1-output', str(phase1), '--output', str(output),
        '--max-units', '16', '--masks-per-image', '16',
    ], cwd=root, check=True, capture_output=True, text=True)
    summary = json.loads((output / 'summary.json').read_text())
    checkpoint = (output / 'masked-predictive-memory.json').read_text()
    assert summary['checks_passed']
    assert summary['data']['training_images'] == 2
    assert summary['data']['heldout_masked_patches'] == 16
    assert 'glioma' not in checkpoint and 'pituitary' not in checkpoint
