"""Research-only OMNIA learning/readout integrity checks."""
from __future__ import annotations
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))

from mycelia.visual_memory import VisualConfig, VisualMycelium, visual_cues
from mycelia.visual_readout import (colony_match_features, centroid_distance_scores,
                                    spatial_pyramid_features)
from omnia_tumor_research import clean_image_splits, read_test_labels
from medical_image_test import LABELS, fit_centroids
from predict_omnia_tumor_research import predict_file


def record(split, label, source, cue, file):
    return dict(split=split, label=label, source_sha256=source,
                cue_sha256=cue, file=file)


def test_duplicate_audit_does_not_need_testing_labels():
    rows = [
        record('Training', 'glioma', 'a', 'ca', 'Training/a.omnia'),
        record('Training', 'glioma', 'a', 'ca', 'Training/a-copy.omnia'),
        record('Training', 'meningioma', 'b', 'cb', 'Training/b.omnia'),
        record('Testing', None, 'a', 'ca', 'Testing/leak.omnia'),
        record('Testing', None, 'c', 'cc', 'Testing/c.omnia'),
        record('Testing', None, 'c', 'cc', 'Testing/c-copy.omnia'),
    ]
    train, test, audit = clean_image_splits(rows)
    assert train.tolist() == [0, 2]
    assert test.tolist() == [4]
    assert audit['removed']['Training_duplicate'] == 1
    assert audit['removed']['Testing_train_overlap'] == 1
    assert audit['removed']['Testing_duplicate'] == 1
    assert audit['test_labels_not_used_for_duplicate_filtering']


def test_colony_match_features_are_fixed_for_a_frozen_population():
    model = VisualMycelium(VisualConfig(size=16, patch_size=4, max_colonies=4,
                                        preprocessing='fixed', multiscale_features=True))
    cues = np.stack([visual_cues(np.full((16, 16), value), multiscale=True)
                     for value in (.2, .8)])
    model.learn(cues[0]); model.learn(cues[1]); model.freeze()
    before = model.state_dict()
    features = colony_match_features(model, cues, batch_size=1)
    repeated = colony_match_features(model, cues, batch_size=2)
    assert features.shape == (2, 4*len(model.colonies)+48)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, repeated)
    np.testing.assert_array_equal(model.patch_scores_batch(cues),
                                  np.stack([model.patch_scores(cue) for cue in cues]))
    assert before == model.state_dict()


def test_spatial_pyramid_features_preserve_location_and_larger_grid():
    model = VisualMycelium(VisualConfig(size=32, patch_size=4, max_colonies=3,
                                        cue_mode='oriented'))
    cues = np.zeros((2, 32, 32, 3), dtype=float)
    cues[..., 1:] = .5
    cues[0, :4, :4, 0] = 1
    cues[1, -4:, -4:, 0] = 1
    model.learn(cues[0]); model.learn(cues[1]); model.freeze()
    features = spatial_pyramid_features(model, cues, levels=(1, 2, 4), batch_size=1)
    assert features.shape == (2, 21*model.colony_count+64*3)
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])
    np.testing.assert_array_equal(features, spatial_pyramid_features(
        model, cues, levels=(1, 2, 4), batch_size=2))


def test_centroid_distances_choose_same_labels_as_existing_readout():
    x = np.asarray([[0., 0.], [.1, 0], [1., 0], [1.1, 0],
                    [2., 0], [2.1, 0], [3., 0], [3.1, 0]])
    y = np.repeat(LABELS, 2)
    model = fit_centroids(x, y)
    scores = centroid_distance_scores(model, x)
    assert scores.shape == (len(x), 4)
    assert np.array_equal(np.asarray(model['labels'])[np.argmin(scores, axis=1)], y)


def test_manifest_testing_labels_are_read_by_separate_postfit_helper(tmp_path):
    manifest = tmp_path/'manifest.csv'
    manifest.write_text('split,label,omnia\nTraining,glioma,dataset/Training/a.omnia\n'
                        'Testing,pituitary,dataset/Testing/b.omnia\n')
    rows = [record('Testing', None, 'b', 'cb', 'Testing/b.omnia')]
    assert read_test_labels(tmp_path, rows, np.asarray([0])).tolist() == ['pituitary']


def test_single_omnia_predictor_matches_frozen_batch_path(tmp_path, monkeypatch):
    # Fake only the SDK container boundary; use the real JPEG decoder, MYCELIA
    # feature path, frozen readout, checkpoint loader, and prediction command.
    pixels = np.tile(np.arange(64, dtype=np.uint8).reshape(8, 8, 1)*4, (1, 1, 3))
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='JPEG', quality=95)
    payload = buffer.getvalue()
    sdk = types.ModuleType('omnia_sdk'); sdk.__path__ = []
    container_module = types.ModuleType('omnia_sdk.container')

    class FakeContainer:
        pixel_codec = 'jpeg'
        num_slices = 1
        def __init__(self, path): self.path = path
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get_slice(self, index):
            assert index == 0
            return np.frombuffer(payload, dtype=np.uint8)

    container_module.OmniaContainer = FakeContainer
    monkeypatch.setitem(sys.modules, 'omnia_sdk', sdk)
    monkeypatch.setitem(sys.modules, 'omnia_sdk.container', container_module)

    image = Image.open(io.BytesIO(payload)).convert('RGB')
    model = VisualMycelium(VisualConfig(max_colonies=4, preprocessing='fixed',
                                        multiscale_features=True, hierarchical_context=True))
    model.learn(model.encode_image(image)); model.freeze()
    model_dir = tmp_path/'model'; model_dir.mkdir()
    memory_path = model_dir/'unsupervised-mycelia-memory.json'
    model.save(memory_path)
    cue = model.encode_image(image)
    feature = colony_match_features(model, cue[None, ...])
    training_features = np.vstack([feature + offset for offset in (0., .1, .2, .3)])
    readout = fit_centroids(training_features, LABELS)
    readout_data = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in readout.items()}
    readout_data.update(schema='mycelia.omnia-colony-centroid.v1',
                        feature_dimension=feature.shape[1],
                        memory_sha256=__import__('hashlib').sha256(memory_path.read_bytes()).hexdigest())
    (model_dir/'external-nearest-centroid-readout.json').write_text(json.dumps(readout_data))
    result = predict_file(tmp_path/'input.omnia', model_dir)
    from mycelia.frozen_visual import FrozenVisualMatcher
    compiled_path = tmp_path/'compiled.npz'
    FrozenVisualMatcher.from_model(model, readout_data['memory_sha256']).save(compiled_path)
    compiled_result = predict_file(tmp_path/'input.omnia', model_dir,
                                   compiled_model=compiled_path)
    assert compiled_result == result
    scores = centroid_distance_scores(readout, feature)
    expected = readout['labels'][int(np.argmin(scores[0]))]
    assert result['predicted_dataset_label'] == expected
    assert len(result['class_distance_scores']) == 4
    assert 'not probabilities' in result['score_interpretation']
