import hashlib
import json

import numpy as np
import pytest

from mycelia.frozen_visual import FrozenVisualMatcher
from mycelia.visual_memory import VisualConfig, VisualMycelium
from mycelia.visual_readout import colony_match_features


def trained_model():
    model = VisualMycelium(VisualConfig(size=16, patch_size=4, max_colonies=6))
    cues = np.random.default_rng(84).random((3, 16, 16, 3))
    for cue in cues:
        model.learn(cue)
    model.freeze()
    return model, cues


def test_compiled_graph_preserves_scores_features_and_reload(tmp_path):
    model, cues = trained_model()
    snapshot = FrozenVisualMatcher.from_model(model, 'a'*64)
    np.testing.assert_array_equal(snapshot.patch_scores_batch(cues), model.patch_scores_batch(cues))
    np.testing.assert_array_equal(colony_match_features(snapshot, cues), colony_match_features(model, cues))
    snapshot.save(tmp_path/'model.npz')
    restored = FrozenVisualMatcher.load(tmp_path/'model.npz')
    np.testing.assert_array_equal(restored.patch_scores_batch(cues), snapshot.patch_scores_batch(cues))
    assert restored.config == model.config
    assert restored.source_memory_sha256 == 'a'*64
    assert restored.patch_scores_batch(cues[:0]).shape == (0, 16, 6)


def test_snapshot_is_independent_and_respects_damage():
    model, cues = trained_model()
    # Existing graph semantics include both retired tissue and uneven routes.
    for node in model.colonies[0].organism.nodes.values():
        node.alive = False
    for edge in model.colonies[1].organism.segments.values():
        edge.alive = False
    snapshot = FrozenVisualMatcher(model.config,
        __import__('mycelia.visual_memory', fromlist=['_graph_score_arrays'])._graph_score_arrays(model.colonies),
        'b'*64)
    expected = model.patch_scores_batch(cues)
    np.testing.assert_array_equal(snapshot.patch_scores_batch(cues), expected)
    for node in model.colonies[2].organism.nodes.values():
        node.receptor_trace = [0., 0., 0.]
    np.testing.assert_array_equal(snapshot.patch_scores_batch(cues), expected)
    assert not np.array_equal(model.patch_scores_batch(cues), expected)


def test_compile_requires_frozen_model():
    model = VisualMycelium(VisualConfig(max_colonies=2))
    with pytest.raises(ValueError, match='frozen'):
        FrozenVisualMatcher.from_model(model, 'c'*64)


def test_corrupt_compiled_archive_rejected(tmp_path):
    model, _ = trained_model()
    snapshot = FrozenVisualMatcher.from_model(model, 'a'*64)
    path = tmp_path/'model.npz'
    snapshot.save(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays['traces'][0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='finite'):
        FrozenVisualMatcher.load(path)


def test_invalid_cues_rejected():
    model, cues = trained_model()
    snapshot = FrozenVisualMatcher.from_model(model, 'a'*64)
    cues[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='cue'):
        snapshot.patch_scores_batch(cues)
