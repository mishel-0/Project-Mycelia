"""Label-free learning evidence, recruitment accounting, and failure integrity."""
from copy import deepcopy
import json

import numpy as np
import pytest

from mycelia.visual_memory import (VisualConfig, VisualMycelium, mask_gray_patch,
                                   prepare_gray, prepare_gray_fixed, visual_cues)


def cue(value, size=8):
    return visual_cues(np.full((size, size), value))


def test_local_learning_remembers_two_distinct_patterns_without_labels():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=4))
    a, b = cue(.1), cue(.9)
    initial = np.mean((m.reconstruct(a)[0]-a)**2)
    m.learn(a); m.learn(b)
    assert len(m.colonies) == 2
    for x in (a, b):
        assert np.mean((m.reconstruct(x)[0]-x)**2) < initial*.01
    m.validate()
    assert 'labels' not in m.state_dict()


def test_already_learned_patches_do_not_spend_or_overwrite():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4))
    x = cue(.2)
    m.learn(x)
    before = [c.state_dict() for c in m.colonies]
    update = m.learn(x)
    assert update['suppressed'] == 4 and update['updates'] == 0
    assert update['synthesis_carbon'] == update['synthesis_energy'] == 0
    assert before == [c.state_dict() for c in m.colonies]
    m.validate()


def test_recruitment_draws_all_initial_resources_and_cannot_exceed_capacity():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=2))
    initial = dict(m.initial_funding)
    m.learn(cue(.1)); m.learn(cue(.9)); r = m.learn(cue(.5))
    assert len(m.colonies) == 2 and r['overflow'] > 0
    for k in initial:
        assert m.remaining_funding[k] == pytest.approx(0, abs=1e-10)
        assert initial[k] == pytest.approx(sum(getattr(c.organism, 'initial_'+k) for c in m.colonies))
    m.validate()


def test_large_bank_can_use_its_last_funded_inoculum_without_rounding_starvation():
    m = VisualMycelium(VisualConfig(size=8, patch_size=8, max_colonies=256))
    rng = np.random.default_rng(491)
    for _ in range(256): m.learn(rng.random((8, 8, 3)))
    assert len(m.colonies) == 256
    assert all(v == 0 for v in m.remaining_funding.values())
    m.validate()


def test_batch_matching_agrees_with_actual_graph_and_reacts_to_erasure():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=4))
    m.learn(cue(.1)); m.learn(cue(.9))
    x = visual_cues(np.random.default_rng(7).random((8, 8)))
    scalar = np.array([[c.distance(p) for c in m.colonies] for p in m.patches(x)])
    np.testing.assert_allclose(m.patch_scores(x), scalar, atol=1e-14)
    for c in m.colonies:
        for n in c.organism.nodes.values():
            n.receptor_trace = []
    assert np.isinf(m.patch_scores(x)).all()
    assert (m.reconstruct(x)[1] == -1).all()


def test_patch_score_batch_matches_repeated_single_image_scores_without_state_change():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=4))
    m.learn(cue(.1)); m.learn(cue(.9))
    images = np.stack([visual_cues(np.full((8, 8), value)) for value in (.2, .7)])
    before = m.state_dict()
    batch = m.patch_scores_batch(images)
    scalar = np.stack([m.patch_scores(image) for image in images])
    np.testing.assert_allclose(batch, scalar, atol=1e-14)
    assert batch.shape == (2, 4, len(m.colonies))
    assert before == m.state_dict()
    with pytest.raises(ValueError):
        m.patch_scores_batch(images[:, :, :, :2])


def test_retired_tissue_cannot_supply_recognition():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4))
    m.learn(cue(.2))
    for c in m.colonies:
        for n in c.organism.nodes.values():
            n.alive = False
        for edge in c.organism.segments.values(): edge.alive = False
    assert (m.reconstruct(cue(.2))[1] == -1).all()


def test_fully_retired_colony_slot_is_recycled_and_remains_resource_accounted():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=1))
    m.learn(cue(.2))
    for node in m.colonies[0].organism.nodes.values(): node.alive = False
    for edge in m.colonies[0].organism.segments.values(): edge.alive = False
    result = m.learn(cue(.8))
    assert result['recycled'] == 1
    assert len(m.colonies) == 1 and m.images_seen == 2
    assert m.summary()['recycled_colonies'] == 1
    assert m.recognize(cue(.8))['match_type'] == 'exact'
    assert m.updates == sum(colony.exposures for colony in m.colonies)
    m.validate()


def test_capacity_consolidates_redundant_motifs_and_reuses_freed_funding(tmp_path):
    config = VisualConfig(size=4, patch_size=4, max_colonies=2,
                          novelty_threshold=.000001, correction_threshold=.0000001,
                          motif_merge_threshold=.03)
    m = VisualMycelium(config)
    images = [visual_cues(np.full((4, 4), value)) for value in (.2, .21, .8)]
    m.learn(images[0]); m.learn(images[1])
    assert len(m.colonies) == 2 and m.colony_support == [1, 1]
    before = m.motif_merges
    result = m.learn(images[2])
    assert result['motif_merges'] == 1 and m.motif_merges == before+1
    assert len(m.colonies) == 2 and all(any(n.alive for n in c.organism.nodes.values()) for c in m.colonies)
    assert m.colony_support == [2, 1]
    assert m.summary()['novelty_overflow'] == 0
    m.validate()
    path = tmp_path/'motif-memory.json'; m.save(path)
    restored = VisualMycelium.load(path)
    assert restored.state_dict() == m.state_dict()


def test_hierarchical_motifs_follow_specialists_through_consolidation():
    config = VisualConfig(size=8, patch_size=4, max_colonies=2,
                          novelty_threshold=.000001, correction_threshold=.0000001,
                          motif_merge_threshold=.03, hierarchical_context=True)
    m = VisualMycelium(config)
    images = [visual_cues(np.full((8, 8), value)) for value in (.2, .21, .8)]
    m.learn(images[0]); m.learn(images[1])
    assert m.context_motifs
    result = m.learn(images[2])
    assert result['motif_merges'] == 1
    assert m.context_motifs and m.summary()['context_motif_connections'] == len(m.context_motifs)
    m.validate()


def test_motif_consolidation_rolls_back_when_image_lesson_fails(monkeypatch):
    m = VisualMycelium(VisualConfig(size=4, patch_size=4, max_colonies=2,
                                    novelty_threshold=.000001, correction_threshold=.0000001,
                                    motif_merge_threshold=.03))
    images = [visual_cues(np.full((4, 4), value)) for value in (.2, .21, .8)]
    m.learn(images[0]); m.learn(images[1])
    before = deepcopy(m.state_dict())
    def fail(_cue): raise ArithmeticError('injected failure after consolidation')
    monkeypatch.setattr(m, '_learn_context', fail)
    with pytest.raises(ArithmeticError): m.learn(images[2])
    assert m.state_dict() == before
    m.validate()


def test_retired_slot_replacement_rolls_back_if_later_part_of_lesson_fails(monkeypatch):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=1))
    m.learn(cue(.2))
    for node in m.colonies[0].organism.nodes.values(): node.alive = False
    for edge in m.colonies[0].organism.segments.values(): edge.alive = False
    before = deepcopy(m.state_dict())
    def fail(_cue): raise ArithmeticError('injected context failure')
    monkeypatch.setattr(m, '_learn_context', fail)
    with pytest.raises(ArithmeticError): m.learn(cue(.8))
    assert m.state_dict() == before
    m.validate()


def test_invalid_image_lesson_is_noop_and_category_arguments_are_rejected():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4))
    before = deepcopy(m.state_dict())
    with pytest.raises(ValueError): m.learn(np.full((8, 8, 3), float('nan')))
    with pytest.raises(TypeError): m.learn(cue(.2), label='glioma')
    assert m.state_dict() == before


def test_checkpoint_resume_freeze_and_read_only_reconstruction(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=3))
    x, y = cue(.15), cue(.85)
    m.learn(x); m.learn(y)
    path = tmp_path/'memory.json'; m.save(path)
    restored = VisualMycelium.load(path)
    m.learn(cue(.45)); restored.learn(cue(.45))
    assert m.state_dict() == restored.state_dict()
    m.freeze(); m.save(path); restored = VisualMycelium.load(path)
    before = deepcopy(restored.state_dict())
    for z in (x, y):
        np.testing.assert_array_equal(m.reconstruct(z)[0], restored.reconstruct(z)[0])
    assert restored.state_dict() == before
    with pytest.raises(RuntimeError): restored.learn(x)


def test_whole_image_failure_rolls_back_recruitment_and_prior_patches(monkeypatch):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=4))
    m.learn(cue(.1))
    before = deepcopy(m.state_dict())
    original = m._recruit
    calls = []
    def fail_second(patch):
        calls.append(1)
        if len(calls) == 2: raise ArithmeticError('injected recruitment failure')
        return original(patch)
    monkeypatch.setattr(m, '_recruit', fail_second)
    x = np.full((8, 8), .5); x[:4, :4] = .12; x[4:, :4] = .9
    with pytest.raises(ArithmeticError): m.learn(visual_cues(x))
    assert m.state_dict() == before
    m.validate()


def test_synthesis_validation_failure_restores_graph_environment_and_exposures(monkeypatch):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=1))
    m.learn(cue(.1)); colony = m.colonies[0]; env = colony.organism.environment
    before = deepcopy(m.state_dict())
    def fail(): raise ArithmeticError('injected validation failure after synthesis')
    monkeypatch.setattr(colony.organism, '_validate', fail)
    with pytest.raises(ArithmeticError): m.learn(cue(.9))
    assert m.state_dict() == before and colony.organism.environment is env


@pytest.mark.parametrize('field,value', [('frozen', 'false'), ('images_seen', True), ('patches_seen', -1), ('updates', 100), ('selection_counts', [True]), ('remaining_funding', {'carbon': -1})])
def test_checkpoint_corruption_is_rejected(tmp_path, field, value):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4)); m.learn(cue(.2))
    state = m.state_dict(); state[field] = value
    path = tmp_path/'bad.json'; path.write_text(json.dumps(state))
    with pytest.raises((ValueError, ArithmeticError)): VisualMycelium.load(path)


@pytest.mark.parametrize('kwargs', [{'size': True}, {'size': 31}, {'patch_size': 2}, {'max_colonies': 0}, {'novelty_threshold': float('nan')}, {'learning_rate': 2}, {'detailed': 'yes'}, {'correction_threshold': .02}, {'motif_merge_threshold': .3}, {'motif_merge_threshold': True}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): VisualConfig(**kwargs)


def test_visual_detail_and_preprocessing_are_finite_and_aspect_preserving():
    source = np.zeros((24, 48)); source[4:20, 4:44] = .7
    prepared = prepare_gray(source, 32)
    assert prepared.shape == (32, 32)
    assert not prepared[:9].any() and prepared[12:20].max() > .6
    x = np.zeros((32, 32)); x[8:24, 8:24] = 1
    detail = visual_cues(x)
    assert detail.shape == (32, 32, 3) and np.isfinite(detail).all()
    assert detail[..., 1].max() > .5 and detail[..., 2].max() > .5


def test_fixed_preprocessing_does_not_use_image_wide_crop_or_contrast_statistics():
    source = np.full((32, 32), .4)
    changed = source.copy(); changed[20:, 20:] = 1
    first, second = prepare_gray_fixed(source), prepare_gray_fixed(changed)
    np.testing.assert_array_equal(first[:20, :20], second[:20, :20])
    np.testing.assert_array_equal(first, source)


def test_multiscale_cues_preserve_intensity_and_add_broad_context():
    y, x = np.mgrid[:32, :32]
    image = np.clip(.5 + .35*np.sin(x*np.pi/3) + .12*np.sin(y*np.pi/11), 0, 1)
    fine = visual_cues(image)
    multi = visual_cues(image, multiscale=True)
    assert multi.shape == fine.shape == (32, 32, 3)
    np.testing.assert_array_equal(multi[..., 0], fine[..., 0])
    assert not np.array_equal(multi[..., 1], fine[..., 1])
    assert not np.array_equal(multi[..., 2], fine[..., 2])
    assert np.isfinite(multi).all() and np.all((multi >= 0) & (multi <= 1))


def test_oriented_visual_cues_preserve_edge_direction():
    vertical = np.zeros((32, 32)); vertical[:, 16:] = 1
    horizontal = vertical.T
    a = visual_cues(vertical, multiscale=True, mode='oriented')
    b = visual_cues(horizontal, multiscale=True, mode='oriented')
    assert a.shape == b.shape == (32, 32, 3)
    assert np.isfinite(a).all() and np.all((a >= 0) & (a <= 1))
    assert np.max(np.abs(a[..., 1]-.5)) > np.max(np.abs(a[..., 2]-.5))
    assert np.max(np.abs(b[..., 2]-.5)) > np.max(np.abs(b[..., 1]-.5))
    np.testing.assert_array_equal(
        a, visual_cues(vertical, multiscale=True, mode='oriented'))
    energy = visual_cues(vertical, multiscale=True, mode='axis_edges')
    assert energy[..., 1].max() > 0
    assert energy[..., 2].max() == 0


def test_masking_hides_target_before_neighboring_multiscale_cues_are_computed():
    image = np.random.default_rng(933).random((32, 32))
    changed = image.copy(); changed[8:16, 16:24] = 1-changed[8:16, 16:24]
    masked = mask_gray_patch(image, 6)
    masked_changed = mask_gray_patch(changed, 6)
    np.testing.assert_array_equal(masked, masked_changed)
    np.testing.assert_array_equal(visual_cues(masked, multiscale=True),
                                  visual_cues(masked_changed, multiscale=True))
    with pytest.raises(ValueError): mask_gray_patch(image, 16)


def test_model_level_image_api_uses_saved_multiscale_encoding():
    gray = np.random.default_rng(712).random((23, 31))
    m = VisualMycelium(VisualConfig(size=16, patch_size=8, max_colonies=16,
                                    multiscale_features=True))
    expected = visual_cues(prepare_gray(gray, 16), multiscale=True)
    np.testing.assert_array_equal(m.encode_image(gray), expected)
    learned = m.learn_image(gray)
    assert learned['episode_stored'] and m.images_seen == 1
    assert m.recognize_image(gray)['match_type'] == 'exact'
    restored, assignments = m.reconstruct_image(gray)
    assert restored.shape == expected.shape and (assignments >= 0).all()
    m.validate()


def test_image_api_and_checkpoint_preserve_fixed_preprocessing(tmp_path):
    source = np.random.default_rng(127).random((19, 27))
    m = VisualMycelium(VisualConfig(size=16, patch_size=8, max_colonies=16,
                                    multiscale_features=True, preprocessing='fixed'))
    expected = visual_cues(prepare_gray_fixed(source, 16), multiscale=True)
    np.testing.assert_array_equal(m.encode_image(source), expected)
    m.learn_image(source)
    path = tmp_path/'fixed.json'; m.save(path)
    restored = VisualMycelium.load(path)
    assert restored.config.preprocessing == 'fixed'
    np.testing.assert_array_equal(restored.encode_image(source), expected)
    assert restored.state_dict() == m.state_dict()


def test_patch_reassembly_preserves_spatial_positions():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=8))
    x = np.zeros((8, 8)); x[:4, :4] = .1; x[:4, 4:] = .35; x[4:, :4] = .65; x[4:, 4:] = .9
    c = visual_cues(x); m.learn(c)
    np.testing.assert_allclose(m.reconstruct(c)[0], c, atol=1e-14)


def test_exact_duplicate_reuses_one_visual_identity_and_near_duplicate_is_distinct():
    m = VisualMycelium(VisualConfig(size=8, patch_size=8, max_colonies=2))
    a = np.full((8, 8, 3), .2)
    near = a.copy(); near[3, 4, :] = .24
    first = m.learn(a)
    duplicate = m.learn(a.copy())
    second = m.learn(near)
    assert first['episode_identity'] == duplicate['episode_identity']
    assert duplicate['episode_duplicate'] and m.summary()['visual_episodes'] == 2
    assert first['episode_identity'] != second['episode_identity']
    assert m.recognize(a)['match_type'] == 'exact'
    assert m.recognize(near)['match_type'] == 'exact'
    perturbed = a.copy(); perturbed[0, 0, 0] += .005
    candidate = m.recognize(perturbed)
    assert candidate['match_type'] == 'perceptual'
    assert candidate['identity'] is None and candidate['candidate_identity'] == first['episode_identity']
    assert candidate['distance'] > 0 and candidate['margin'] is not None


def test_seen_identity_recognition_is_256_of_256_and_checkpoint_safe(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=8, max_colonies=8, max_episodes=300))
    rng = np.random.default_rng(20261004)
    images = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8).astype(float)/255 for _ in range(256)]
    identities = [m.learn(image)['episode_identity'] for image in images]
    matches = [m.recognize(image) for image in images]
    assert all(result['match_type'] == 'exact' and result['identity'] == identity
               for result, identity in zip(matches, identities))
    assert len(set(identities)) == 256
    m.freeze(); path = tmp_path/'episodic-memory.json'; m.save(path)
    restored = VisualMycelium.load(path)
    assert restored.summary()['visual_episodes'] == 256
    assert [restored.recognize(image)['identity'] for image in images] == identities
    assert restored.state_dict() == m.state_dict()


def test_episode_capacity_abstains_for_unstored_novel_image():
    m = VisualMycelium(VisualConfig(size=8, patch_size=8, max_colonies=2, max_episodes=1))
    first, second = cue(.1), cue(.9)
    stored = m.learn(first)
    overflow = m.learn(second)
    assert stored['episode_stored'] and not overflow['episode_stored']
    assert overflow['episode_identity'] is None
    assert m.recognize(second)['match_type'] == 'novel'
    assert m.summary()['visual_episodes'] == 1


def test_visual_v1_checkpoint_loads_with_empty_episode_memory(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4)); m.learn(cue(.2))
    state = m.state_dict(); state['schema'] = 'mycelia-visual-1'
    state.pop('episodes'); state.pop('next_episode_id')
    state['config'].pop('max_episodes'); state['config'].pop('recognition_threshold')
    path = tmp_path/'v1.json'; path.write_text(json.dumps(state))
    restored = VisualMycelium.load(path)
    assert restored.summary()['visual_episodes'] == 0
    assert restored.recognize(cue(.2))['match_type'] == 'novel'


def test_visual_v3_checkpoint_migrates_context_without_predictive_feedback(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4)); m.learn(cue(.2))
    state = m.state_dict(); state['schema'] = 'mycelia-visual-3'
    state.pop('predictive_feedback_patches'); state.pop('predictive_feedback_error')
    state['config'].pop('predictive_error_scale')
    path = tmp_path/'v3.json'; path.write_text(json.dumps(state))
    restored = VisualMycelium.load(path)
    assert restored.context_associations == m.context_associations
    assert restored.predictive_feedback_patches == 0
    assert restored.predictive_feedback_error == 0
    restored.validate()


def test_visual_v4_checkpoint_migrates_without_motif_accounting(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4)); m.learn(cue(.2))
    state = m.state_dict(); state['schema'] = 'mycelia-visual-4'
    state.pop('motif_merges'); state.pop('colony_support')
    state['config'].pop('motif_merge_threshold')
    path = tmp_path/'v4.json'; path.write_text(json.dumps(state))
    restored = VisualMycelium.load(path)
    assert restored.motif_merges == 0
    assert restored.colony_support == [0]*len(restored.colonies)
    assert restored.config.motif_merge_threshold == 0
    restored.validate()


def test_visual_v5_checkpoint_defaults_missing_preprocessing_to_legacy_mode(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4)); m.learn(cue(.2))
    state = m.state_dict(); state['schema'] = 'mycelia-visual-5'
    state['config'].pop('preprocessing')
    path = tmp_path/'v5.json'; path.write_text(json.dumps(state))
    restored = VisualMycelium.load(path)
    assert restored.config.preprocessing == 'foreground'
    assert restored.context_associations == m.context_associations
    restored.validate()


def test_episode_template_corruption_is_rejected(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=8)); m.learn(cue(.2))
    state = m.state_dict(); state['episodes'][0]['template'] = 'not base64'
    path = tmp_path/'corrupt-episode.json'; path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        VisualMycelium.load(path)


def test_masked_patch_prediction_ignores_target_and_returns_stored_candidate():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=3))
    rng = np.random.default_rng(711)
    a = visual_cues(rng.random((8, 8)))
    b = visual_cues(rng.random((8, 8)))
    m.learn(a); m.learn(b)
    original = m.predict_masked_patch(a, 0)
    changed = a.copy(); changed[:4, :4, :] = 1-changed[:4, :4, :]
    hidden_changed = m.predict_masked_patch(changed, 0)
    np.testing.assert_array_equal(original['predicted_patch'], hidden_changed['predicted_patch'])
    assert original['candidate_identity'] == hidden_changed['candidate_identity']
    assert original['context_rmse'] == hidden_changed['context_rmse']
    with pytest.raises(ValueError): m.predict_masked_patch(a, 4)


def test_masked_patch_prediction_requires_episode_memory():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4))
    with pytest.raises(RuntimeError): m.predict_masked_patch(cue(.2), 0)


def test_local_colony_context_learns_without_episode_lookup_and_masks_target():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=2))
    x = cue(.2)
    m.learn(x)
    assert m.summary()['context_connections'] >= 4
    before = m.predict_masked_patch_from_colonies(x, 0)
    assert before['predicted_patch'] is not None and before['context_sources'] > 0
    changed = x.copy(); changed[:4, :4, :] = 1-changed[:4, :4, :]
    after = m.predict_masked_patch_from_colonies(changed, 0)
    np.testing.assert_array_equal(before['predicted_patch'], after['predicted_patch'])
    assert before['candidate_probabilities'] == after['candidate_probabilities']
    # Remove every episodic record: the colony predictor remains operational.
    m.episodes.clear(); m._episode_index.clear(); m.next_episode_id = 0
    assert m.predict_masked_patch_from_colonies(x, 0)['predicted_patch'] is not None
    m.validate()


def test_pair_conditioned_motif_context_learns_and_does_not_use_hidden_target(tmp_path):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=2,
                                    hierarchical_context=True))
    a = cue(.2); m.learn(a)
    assert m.context_motifs and m.context_motif_updates > 0
    visible = np.zeros(4, dtype=int); visible[0] = visible[1] = visible[2] = 0; visible[3] = -1
    before = m.predict_masked_patch_from_colonies(a, 3, patch_assignments=visible,
                                                  context_order=2)
    changed = a.copy(); changed[4:, 4:, :] = .95
    after = m.predict_masked_patch_from_colonies(changed, 3, patch_assignments=visible,
                                                 context_order=2)
    assert before['context_order_used'] == 2
    np.testing.assert_array_equal(before['predicted_patch'], after['predicted_patch'])
    assert before['candidate_probabilities'] == after['candidate_probabilities']
    path = tmp_path/'motifs.json'; m.save(path)
    state = m.state_dict(); restored = VisualMycelium.load(path)
    assert restored.context_motifs == m.context_motifs
    assert restored.state_dict() == state
    restored.validate()


def test_pair_motif_state_rolls_back_after_late_image_learning_failure(monkeypatch):
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=3,
                                    hierarchical_context=True))
    m.learn(cue(.2))
    before = deepcopy(m.state_dict())
    original = m._learn_context
    def fail_after_motif_update(value):
        original(value)
        raise ArithmeticError('injected failure after motif learning')
    monkeypatch.setattr(m, '_learn_context', fail_after_motif_update)
    with pytest.raises(ArithmeticError):
        m.learn(cue(.3))
    assert m.state_dict() == before
    m.validate()


def test_familiar_patch_with_novel_neighbor_relation_triggers_predictive_plasticity():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=2,
                                    predictive_feedback=True))
    a = np.full((8, 8, 3), .2)
    b = np.full((8, 8, 3), .8)
    m.learn(a); m.learn(b)
    # Every local patch is already familiar; only their spatial arrangement is
    # new. The prior A->A and B->B transitions predict the wrong neighbors.
    mixed = np.empty_like(a)
    mixed[:4, :4] = a[:4, :4]; mixed[:4, 4:] = b[:4, 4:]
    mixed[4:, :4] = a[4:, :4]; mixed[4:, 4:] = b[4:, 4:]
    result = m.learn(mixed)
    assert result['mean_predictive_error'] is not None
    assert result['mean_predictive_error'] > m.config.predictive_error_scale
    assert result['predictive_feedback_patches'] == 4
    assert result['updates'] > 0 and result['suppressed'] < 4
    assert m.summary()['predictive_feedback_patches'] >= 4
    m.validate()


def test_predictive_feedback_ablation_keeps_unsupervised_context_learning():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=2,
                                    predictive_feedback=False))
    m.learn(cue(.2)); result = m.learn(cue(.8))
    assert result['predictive_feedback_patches'] == 0
    assert result['mean_predictive_error'] is None
    assert m.summary()['context_connections'] > 0
    assert m.summary()['predictive_feedback_patches'] == 0
    m.validate()


def test_error_feedback_improves_an_unfamiliar_patch_with_no_extra_capacity():
    m = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=1))
    m.learn(cue(.1)); x = cue(.5)
    before = np.mean((m.reconstruct(x)[0]-x)**2)
    for _ in range(5): m.learn(x)
    assert np.mean((m.reconstruct(x)[0]-x)**2) < before*.7
    assert m.summary()['max_absolute_budget_error'] < 1e-8


def test_context_totals_tolerate_float_drift_at_large_magnitude_but_not_real_drift():
    model = VisualMycelium(VisualConfig(size=8, patch_size=4, max_colonies=4))
    for value in (.1, .5, .9):
        model.learn(visual_cues(np.linspace(0, value, 64).reshape(8, 8)))
    assert model._context_row_totals
    row = next(iter(model._context_row_totals))
    targets = model._context_row_targets[row]
    for target in targets:
        targets[target] *= 1e5
        model.context_associations[(row[0], row[1], target)] = targets[target]
    model._context_row_totals[row] = sum(targets.values())*(1+1e-12)
    model.context_updates = int(sum(model.context_associations.values()))+1
    model.validate()
    model._context_row_totals[row] *= 1.01
    with pytest.raises(ValueError, match='row totals'):
        model.validate()
