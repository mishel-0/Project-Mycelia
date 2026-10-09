from types import SimpleNamespace

import numpy as np

from mycelia.volume_context import DirectionalVolumeContext, blend_predictions


def fake_model(prototypes):
    colonies = []
    for value in prototypes:
        nodes = {i: SimpleNamespace(alive=True, receptor_trace=[value] * 3)
                 for i in range(9)}
        colonies.append(SimpleNamespace(organism=SimpleNamespace(nodes=nodes)))
    return SimpleNamespace(config=SimpleNamespace(patch_size=3), colonies=colonies)


def test_routes_predict_across_consecutive_slices_and_ignore_target_assignment():
    memory = DirectionalVolumeContext().fit({'p1': {
        1: np.array([[0]]),
        2: np.array([[1]]),
    }})
    # Deliberately make the masked target assignment wrong. Prediction is from
    # the adjacent slice's colony and the learned +z route only.
    query = {'p1': {1: np.array([[0]]), 2: np.array([[0]])}}
    result = memory.predict(query, 'p1', 2, 0, 0, fake_model([0.1, 0.8]))
    assert result['context_sources'] == 1
    assert result['candidate_probabilities'] == {1: 1.0}
    assert np.allclose(result['predicted_patch'], 0.8)


def test_missing_slice_index_does_not_create_a_through_slice_route():
    memory = DirectionalVolumeContext().fit({'p1': {
        1: np.array([[0]]),
        3: np.array([[1]]),
    }})
    assert memory.updates == 0
    assert memory.route_count == 0


def test_patient_boundaries_never_create_cross_patient_routes():
    memory = DirectionalVolumeContext().fit({
        'p1': {1: np.array([[0]])},
        'p2': {2: np.array([[1]])},
    })
    assert memory.updates == 0
    assert memory.route_count == 0


def test_training_route_is_available_only_from_its_learned_source_direction():
    memory = DirectionalVolumeContext().fit({'p1': {
        1: np.array([[0]]),
        2: np.array([[1]]),
    }})
    # The route from slice 1 toward slice 2 is +z; reversing it must not be
    # silently treated as the same route.
    reverse = {'p1': {1: np.array([[0]]), 2: np.array([[0]])}}
    result = memory.predict(reverse, 'p1', 1, 0, 0, fake_model([0.1, 0.8]))
    assert result['context_sources'] == 0
    assert result['predicted_patch'] is None


def test_blend_uses_training_selected_weight_and_falls_back_if_no_route():
    route = np.full((3, 3, 3), .8)
    local = np.full((3, 3, 3), .2)
    assert np.allclose(blend_predictions(route, local, .25), .35)
    assert np.allclose(blend_predictions(None, local, .9), local)


def test_blend_rejects_invalid_weight_or_shape():
    local = np.zeros((3, 3, 3))
    with np.testing.assert_raises(ValueError):
        blend_predictions(local, local, float('nan'))
    with np.testing.assert_raises(ValueError):
        blend_predictions(np.zeros((2, 2, 3)), local, .5)
