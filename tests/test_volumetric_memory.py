import numpy as np

from mycelia.volumetric_memory import GroupedPredictiveMemory, VolumetricMemoryConfig, VolumetricMycelium


def training_volume():
    volume = np.zeros((4, 1, 1, 2), dtype=float)
    volume[:, 0, 0, :] = [[.1,.7],[.2,.8],[.1,.7],[.2,.8]]
    return volume


def test_competitive_colonies_and_six_direction_routes_learn_without_labels():
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01))
    model.fit([training_volume()], seed=7)
    summary = model.validate()
    assert summary['images_seen'] == 1
    assert summary['patches_seen'] == 4
    assert 1 <= summary['colonies'] <= 8
    assert summary['route_updates'] > 0
    assert summary['routes'] > 0


def test_masked_prediction_is_invariant_to_hidden_center_modality():
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01))
    model.fit([training_volume()], seed=7)
    query = training_volume().copy()
    first = model.predict_masked(query, (1,0,0), 0)['prediction']
    query[1,0,0,0] = .99
    second = model.predict_masked(query, (1,0,0), 0)['prediction']
    assert first == second


def test_route_learning_does_not_wrap_across_volume_boundaries_or_gaps():
    volume = np.zeros((1,1,3,2), dtype=float)
    volume[0,0,0] = [.2,.8]
    volume[0,0,2] = [.8,.2]
    model = VolumetricMycelium().fit([volume])
    assert model.route_updates == 0
    assert model.route_count == 0


def test_volume_predictions_cover_active_patches_and_preserve_inactive_background():
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01))
    volume = training_volume()
    model.fit([volume], seed=1)
    query = np.zeros((4,1,2,2), dtype=float)
    query[:,0,0,:] = volume[:,0,0,:]
    result = model.predict_volume(query)
    assert result['active_patches'] == 4
    assert result['predictions'].shape == query.shape
    assert np.all(np.isnan(result['predictions'][:,0,1,:]))
    assert np.all(np.isfinite(result['predictions'][:,0,0,:]))


def test_colony_local_predictor_uses_context_without_reading_hidden_center():
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01))
    volume = training_volume()
    model.fit([volume], seed=3)
    before = model.predict_local_volume(volume)['predictions'][1, 0, 0, 0]
    changed = volume.copy()
    changed[1, 0, 0, 0] = .99
    after = model.predict_local_volume(changed)['predictions'][1, 0, 0, 0]
    assert before == after
    assert np.isfinite(before)


def test_local_predictive_associations_round_trip_with_colony_memory(tmp_path):
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01))
    volume = training_volume()
    model.fit([volume], seed=11)
    path = tmp_path / 'predictive-memory.npz'
    model.save(path)
    restored = VolumetricMycelium.load(path)
    assert np.array_equal(restored.predictive_support, model.predictive_support)
    assert np.allclose(restored.predict_local_volume(volume)['predictions'],
                       model.predict_local_volume(volume)['predictions'], equal_nan=True)


def test_neighbor_route_votes_change_the_selected_local_visual_specialist():
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01))
    volume = training_volume()
    model.fit([volume], seed=3)
    masked_assignments = model.assignments(volume, masked_coordinate=(1,0,0), masked_channel=0)
    center_specialist = int(masked_assignments[1,0,0])
    target_specialist = (center_specialist + 1) % model.colony_count
    model.predictive_route_rows[0] = {
        (int(masked_assignments[0,0,0]), 0): {target_specialist: 5.0},
        (int(masked_assignments[2,0,0]), 1): {target_specialist: 5.0},
    }
    model.predictive_coefficients[0, target_specialist, -1] += .2
    plain = model.predict_local_volume(volume, route_weight=0.0)['predictions'][1, 0, 0, 0]
    routed = model.predict_local_volume(volume, route_weight=1.0)['predictions'][1, 0, 0, 0]
    assert not np.isclose(plain, routed)
    changed = volume.copy()
    changed[1, 0, 0, 0] = .99
    hidden = model.predict_local_volume(changed, route_weight=1.0)['predictions'][1, 0, 0, 0]
    assert hidden == routed


def test_streaming_update_preserves_prior_memory_and_checkpoint_resume(tmp_path):
    config = VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01)
    first = training_volume()
    second = np.zeros_like(first)
    second[:, 0, 0, :] = [[.7,.3],[.8,.2],[.7,.3],[.8,.2]]
    model = VolumetricMycelium(config).fit([first], seed=5)
    initial_state = model.validate()
    initial_colonies = model.colony_count
    initial_routes = dict(model.routes)
    checkpoint = tmp_path / 'streaming-memory.npz'
    model.save(checkpoint)
    resumed = VolumetricMycelium.load(checkpoint)
    model.partial_fit(second)
    resumed.partial_fit(second)
    state = model.validate()
    assert state['images_seen'] == 2
    assert state['patches_seen'] == initial_state['patches_seen'] * 2
    assert state['colonies'] > initial_colonies
    assert all(model.routes[key] >= weight for key, weight in initial_routes.items())
    assert resumed.validate() == state
    assert np.allclose(resumed.centers, model.centers)
    assert np.allclose(resumed.predict_local_volume(second)['predictions'],
                       model.predict_local_volume(second)['predictions'], equal_nan=True)


def test_streaming_novelty_adapts_when_colony_capacity_is_full():
    config = VolumetricMemoryConfig(max_colonies=2, novelty_threshold=.01)
    first = training_volume()
    second = np.zeros_like(first)
    second[:, 0, 0, :] = [.9,.1]
    model = VolumetricMycelium(config).fit([first], seed=3)
    before = model.centers.copy()
    model.partial_fit(second)
    assert model.colony_count == config.max_colonies
    assert model.validate()['images_seen'] == 2
    assert not np.array_equal(model.centers, before)


def test_topology_only_fit_can_be_used_by_grouped_predictive_memory(tmp_path):
    volume = training_volume()
    colonies = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=4)).fit(
        [volume], seed=4, fit_predictors=False)
    state = colonies.validate()
    assert state['predictive_maps_enabled'] is False
    with np.testing.assert_raises(RuntimeError):
        colonies.predict_volume(volume)
    predictor = GroupedPredictiveMemory(colonies, ((0,), (1,))).fit([volume])
    assert np.isfinite(predictor.predict_volume(volume)['predictions'][..., 0]).any()

    checkpoint = tmp_path / 'topology-only.npz'
    colonies.save(checkpoint)
    restored = VolumetricMycelium.load(checkpoint)
    assert restored.validate()['predictive_maps_enabled'] is False


def test_grouped_predictor_can_use_compact_context_to_predict_full_channel_group():
    volume = np.random.default_rng(17).normal(size=(4, 2, 2, 16))
    volume = np.clip(volume, -1, 1)
    colonies = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=4)).fit(
        [volume], seed=12, fit_predictors=False)
    context = (0, 1, 2, 3, 8, 9, 10, 11)
    predictor = GroupedPredictiveMemory(
        colonies, (tuple(range(8)), tuple(range(8, 16))),
        context_scales=(1,), context_feature_channels=context).fit([volume])
    assert predictor.global_coefficients[0].shape[0] == 1 + 6*len(context) + 4
    result = predictor.predict_volume(volume, local_weights=0)
    assert np.isfinite(result['predictions'][..., :8]).any()


def test_grouped_predictive_memory_hides_every_center_cue_in_target_group():
    volume = np.zeros((4, 1, 1, 4), dtype=float)
    volume[:, 0, 0] = [[.1,.2,.3,.4],[.2,.4,.6,.8],[.1,.2,.3,.4],[.2,.4,.6,.8]]
    colonies = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01)).fit([volume])
    predictor = GroupedPredictiveMemory(colonies, ((0,1),(2,3))).fit([volume])
    before_result = predictor.predict_volume(volume, local_weights=.5)
    before = before_result['predictions'][1,0,0,0:2].copy()
    changed = volume.copy()
    changed[1,0,0,0:2] = [.95,.05]
    changed_result = predictor.predict_volume(changed, local_weights=.5)
    after = changed_result['predictions'][1,0,0,0:2]
    assert np.array_equal(before, after)
    assert before_result['active_patches_by_group'] == changed_result['active_patches_by_group']
    assert np.all(np.isfinite(before))


def test_grouped_predictor_accepts_per_colony_specialist_influence():
    volume = training_volume()
    colonies = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=4)).fit([volume], seed=4)
    predictor = GroupedPredictiveMemory(colonies, ((0,), (1,))).fit([volume])
    shared = predictor.predict_volume(volume, local_weights=0.0)
    local = predictor.predict_volume(volume, local_weights=1.0)
    assert np.allclose(shared['predictions'], shared['global_predictions'], equal_nan=True)
    assert np.allclose(local['predictions'], local['local_predictions'], equal_nan=True)

    assignments = shared['colony_assignments']
    weights = np.zeros((2, 4), dtype=float)
    group, colony = 0, int(assignments[..., 0][assignments[..., 0] >= 0][0])
    weights[group, colony] = 1.0
    mixed = predictor.predict_volume(volume, local_weights=weights)
    active = np.isfinite(shared['predictions'][..., 0])
    target_mask = (assignments[..., group] == colony) & active
    other_mask = (assignments[..., group] >= 0) & ~target_mask & active
    assert np.allclose(mixed['predictions'][..., 0][target_mask],
                       local['predictions'][..., 0][target_mask])
    assert np.allclose(mixed['predictions'][..., 0][other_mask],
                       shared['predictions'][..., 0][other_mask])


def test_grouped_context_includes_second_radius_without_wrapping():
    volume = np.zeros((3, 1, 1, 2), dtype=float)
    volume[0, 0, 0, 0] = .7
    context = GroupedPredictiveMemory.context_matrix(volume, (0,))
    center = np.ravel_multi_index((2, 0, 0), volume.shape[:3])
    # Six radius-one vectors precede the radius-two vectors; -z reaches z=0.
    assert context.shape == (3, 6*2*2+(2-1)+1)
    assert np.isclose(context[center, 6*2], .7)
    # The opposite face remains zero instead of wrapping to the first slice.
    first = np.ravel_multi_index((0, 0, 0), volume.shape[:3])
    assert context[first, 6*2:12*2].sum() == 0


def test_grouped_predictor_requires_fit_before_inference():
    colonies = VolumetricMycelium(VolumetricMemoryConfig()).fit([training_volume()])
    predictor = GroupedPredictiveMemory(colonies, ((0,),))
    with np.testing.assert_raises(RuntimeError):
        predictor.predict_volume(training_volume())


def test_grouped_predictor_rejects_boolean_specialist_weight():
    volume = training_volume()
    colonies = VolumetricMycelium(VolumetricMemoryConfig()).fit([volume])
    predictor = GroupedPredictiveMemory(colonies, ((0,),)).fit([volume])
    with np.testing.assert_raises(ValueError):
        predictor.predict_volume(volume, local_weights=True)


def test_signed_visual_coefficients_are_active_without_marking_zero_background_active():
    volume = np.zeros((3, 1, 1, 3), dtype=float)
    volume[1, 0, 0] = [-.2, .1, -.4]
    volume[2, 0, 0] = [-.4, .2, -.1]
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=4)).fit([volume])
    assert model.validate()['patches_seen'] == 2
    assert model.predict_volume(volume)['active_patches'] == 2


def test_invalid_memory_parameters_fail_fast():
    with np.testing.assert_raises(ValueError):
        VolumetricMemoryConfig(max_colonies=1)


def test_colony_and_route_state_round_trips_without_pickle(tmp_path):
    model = VolumetricMycelium(VolumetricMemoryConfig(max_colonies=8, novelty_threshold=.01))
    volume = training_volume()
    model.fit([volume], seed=11)
    path = tmp_path / 'memory.npz'
    model.save(path)
    restored = VolumetricMycelium.load(path)
    assert restored.validate() == model.validate()
    assert np.array_equal(restored.centers, model.centers)
    assert np.array_equal(restored.support, model.support)
    assert np.array_equal(restored.predict_volume(volume)['predictions'],
                          model.predict_volume(volume)['predictions'])
