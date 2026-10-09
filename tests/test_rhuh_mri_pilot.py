from pathlib import Path
import importlib.util
import csv
from types import SimpleNamespace

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "rhuh_mri_pilot", Path(__file__).parents[1] / "tools" / "rhuh_mri_pilot.py"
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_rhuh_cv", Path(__file__).parents[1] / "tools" / "summarize_rhuh_cv.py"
)
summary_tools = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(summary_tools)


def test_masked_context_excludes_center_target_and_does_not_wrap_edges():
    grid = np.zeros((4, 3, 2, 5), dtype=np.float32)
    grid[0, 0, 0] = [7, 2, 3, 4, 5]
    pairs = pilot.context_features(grid)
    center_index = np.ravel_multi_index((0, 0, 0), grid.shape[:3])
    # Target modality 0 is absent from its own center predictors.
    assert np.array_equal(pairs[0][0][center_index, -4:], [2, 3, 4, 5])
    # At the opposite x face, a periodic wrap would leak the corner patch;
    # ordinary spatial boundaries must be zero-padded instead.
    opposite_index = np.ravel_multi_index((3, 0, 0), grid.shape[:3])
    assert np.all(pairs[0][0][opposite_index, :30] == 0)


def test_fixed_3d_patch_descriptors_detect_texture_beyond_patch_mean():
    volume = np.full((2, 2, 4), .5, dtype=np.float32)
    volume[:, :, 2:] = np.asarray([0.0, 1.0], dtype=np.float32)[None, None, :]
    descriptors = pilot.visual_patch_descriptors(volume, patch=2)
    assert descriptors.shape == (1, 1, 2, 3)
    assert np.isclose(descriptors[0, 0, 0, 0], descriptors[0, 0, 1, 0])
    assert descriptors[0, 0, 1, 1] > descriptors[0, 0, 0, 1]


def test_dct8_features_preserve_patch_mean_and_signed_spatial_structure():
    flat = pilot.visual_patch_dct8(np.full((4, 4, 4), .25, dtype=np.float32))
    assert flat.shape == (1, 1, 1, 8)
    assert np.isclose(flat[0, 0, 0, 0], .25)
    assert np.allclose(flat[0, 0, 0, 1:], 0, atol=1e-6)

    ramp = np.broadcast_to(np.arange(4, dtype=np.float32)[None, None, :]/3,
                           (4, 4, 4)).copy()
    encoded = pilot.visual_patch_dct8(ramp)[0, 0, 0]
    assert np.isclose(encoded[0], .5)
    assert abs(encoded[3]) > .05  # signed x-direction frequency survives encoding
    assert np.all(np.abs(encoded) <= 1)


def test_dct64_features_retain_every_patch_mode_and_reconstruct_source():
    rng = np.random.default_rng(229)
    source = rng.uniform(0, 1, (4, 4, 4)).astype(np.float32)
    encoded = pilot.visual_patch_dct64(source)
    assert encoded.shape == (1, 1, 1, 64)
    positions = np.arange(4, dtype=np.float64)
    frequencies = np.arange(4, dtype=np.float64)[:, None]
    basis = np.sqrt(.5)*np.cos(np.pi*(2*positions+1)*frequencies/8)
    basis[0] = .5
    restored = np.zeros((4, 4, 4), dtype=float)
    for value, (z, y, x) in zip(encoded.reshape(-1), pilot._DCT64_FREQUENCIES):
        kernel = np.einsum('i,j,k->ijk', basis[z], basis[y], basis[x])
        restored += float(value)*np.abs(kernel).sum()*kernel
    assert np.allclose(restored, source, atol=1e-6)


def test_dct8_inverse_projection_mse_uses_parseval_scaling():
    coefficient_mses = np.zeros(8)
    coefficient_mses[0] = .01  # normalized DC equals source-patch mean
    assert np.isclose(summary_tools.dct8_projection_mse(coefficient_mses), .01)


def test_dct8_visit_loader_keeps_one_feature_vector_per_patch():
    nib = __import__('pytest').importorskip('nibabel')
    rng = np.random.default_rng(42)
    files = {}
    with __import__('tempfile').TemporaryDirectory() as directory:
        root = Path(directory)
        for index, modality in enumerate(('t1', 't1ce', 't2', 'flair')):
            image = nib.Nifti1Image(rng.uniform(.1, 1, (16, 16, 16)).astype(np.float32), np.eye(4))
            files[modality] = root / f'{modality}.nii.gz'
            nib.save(image, str(files[modality]))
        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        mask[5:8, 5:8, 5:8] = 3
        files[pilot.SEG] = root / 'mask.nii.gz'
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(files[pilot.SEG]))
        _, visual, _, _, _ = pilot.load_visit(
            files, factor=1, patch=4, modalities=('t1', 't1ce', 't2', 'flair'),
            visual_representation='dct8')
    assert visual.shape[1] == 32
    assert np.all(np.isfinite(visual))


def test_patient_visit_scan_groups_complete_visits_only(tmp_path):
    patient = tmp_path / "RHUH-0099"
    for visit in ("0", "1"):
        directory = patient / visit
        directory.mkdir(parents=True)
        stem = f"RHUH-0099_{visit}"
        for modality in pilot.MODALITIES:
            (directory / f"{stem}_{modality}.nii.gz").touch()
        if visit == "0":
            (directory / f"{stem}_segmentations.nii.gz").touch()
        else:
            (directory / "segmentation.nii.gz").touch()
    result = pilot.patient_visits(tmp_path)
    assert list(result) == ["RHUH-0099"]
    assert [visit for visit, _ in result["RHUH-0099"]] == ["0", "1"]


def test_patient_folds_are_disjoint_and_cover_every_person():
    patients = [f"RHUH-{i:04d}" for i in range(1, 41)]
    held_out = []
    for fold in range(5):
        train, test = pilot.patient_split(patients, seed=27, fold=fold, folds=5)
        assert not train.intersection(test)
        assert len(test) == 8
        held_out.extend(test)
    assert sorted(held_out) == sorted(patients)


def test_native_route_blend_calibrates_only_with_outer_training_patients():
    train_patients = {f"P{i}" for i in range(6)}
    visits = []
    coords = np.indices((2, 2, 2)).reshape(3, -1).T
    for i, patient in enumerate(sorted(train_patients)):
        grid = np.full((2, 2, 2, 3), 0.1 + i * 0.01, dtype=np.float32)
        visits.append((patient, "v0", grid[tuple(coords.T)], np.zeros(len(coords)),
                       coords, (2, 2, 2)))
    args = SimpleNamespace(clusters=4, seed=9, fold=0, folds=2)
    weight, audit = pilot.calibrate_native_route_weight(visits, train_patients, args)
    assert weight in (0.0, 0.25, 0.5, 0.75, 1.0)
    inner_train = set(audit["inner_train_patients"])
    validation = set(audit["validation_patients"])
    assert inner_train.isdisjoint(validation)
    assert inner_train | validation == train_patients
    assert set(audit["patient_macro_mse_by_route_weight"]) == {"0.0", "0.25", "0.5", "0.75", "1.0"}
    per_modality_weight, route_audit = pilot.calibrate_local_predictive_route_weight(
        visits, train_patients, args)
    assert len(per_modality_weight) == 3
    assert set(per_modality_weight) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    route_train = set(route_audit["inner_train_patients"])
    route_validation = set(route_audit["validation_patients"])
    assert route_train.isdisjoint(route_validation)
    assert route_train | route_validation == train_patients
    assert set(route_audit["patient_macro_mse_by_route_weight"]) == {"0", "1", "2"}


def test_grouped_specialist_influence_calibrates_on_inner_patients_only():
    rng = np.random.default_rng(33)
    visits = []
    coords = np.indices((4, 2, 2)).reshape(3, -1).T
    for index in range(5):
        visual = rng.uniform(.1, .9, (len(coords), 4)).astype(np.float32)
        visits.append((f"P{index}", "v0", np.zeros((len(coords), 2)),
                       np.full(len(coords), 3), coords, (4, 2, 2), visual))
    args = SimpleNamespace(seed=17, fold=1, folds=5, clusters=4, native_streaming=False)
    weights, audit = pilot.calibrate_grouped_specialist_influence(
        visits, {f"P{i}" for i in range(5)}, args, ((0, 1), (2, 3)))
    assert weights.shape == (2, args.clusters)
    assert set(np.unique(weights)) <= {0.0, .25, .5, .75, 1.0}
    inner_train = set(audit['inner_train_patients'])
    validation = set(audit['validation_patients'])
    assert inner_train.isdisjoint(validation)
    assert inner_train | validation == {f"P{i}" for i in range(5)}
    assert audit['target_group_activity_uses_visible_channels_only'] is True
    assert audit['labels_used_for_calibration'] is False
    # Only one inner-validation patient exists, so no colony may specialize.
    assert np.all(weights == 0)
    assert audit['minimum_patients'] == 2
    assert all(
        colony['fallback_reason'] in {
            'insufficient_total_patches', 'insufficient_independent_patients'
        }
        for group in audit['colony_calibration_by_group'].values()
        for colony in group.values()
    )


def test_colony_specialist_weights_follow_visual_prototypes_across_refits():
    source = np.asarray([[0., 0.], [1., 0.], [0., 1.]])
    target = source[[2, 0, 1]]
    source_weights = np.asarray([[.25, .5, .75], [1., 0., .25]])
    aligned, audit = pilot.align_colony_specialist_weights(
        source_weights, source, target, capacity=4)
    assert np.allclose(aligned[:, :3], source_weights[:, [2, 0, 1]])
    assert np.all(aligned[:, 3] == 0)
    assert audit['unmatched_outer_weight'] == 0
    assert max(row['prototype_rms_distance'] for row in audit['matches']) == 0


def test_indexed_public_mirror_uses_four_verified_sequence_names_and_t1ce_mask(tmp_path):
    rows = []
    sequences = {"t1": "T1", "t1ce": "T1C", "t2": "T2", "flair": "FLAIR"}
    for modality, sequence in sequences.items():
        image_id = f"image-{modality}"
        for kind, directory in (("images", "images"), ("masks", "masks")):
            rows.append({"kind": kind, "id": image_id, "subject": "RHUH-0001",
                         "timepoint": "preoperative", "sequence": sequence,
                         "path": f"{directory}/{image_id}.nii.gz"})
    with (tmp_path / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    visits = pilot.patient_visits(tmp_path, tuple(sequences))
    assert list(visits) == ["RHUH-0001"]
    visit_id, files = visits["RHUH-0001"][0]
    assert visit_id == "0"
    assert set(files) == {"t1", "t1ce", "t2", "flair", pilot.SEG}
    assert files["t1ce"].name == "image-t1ce.nii.gz"
    assert files[pilot.SEG] == tmp_path / "masks/image-t1ce.nii.gz"


def test_discrete_mask_is_resampled_to_reference_orientation_without_label_mixing():
    nib = __import__('pytest').importorskip('nibabel')
    shape = (16, 16, 16)
    reference_affine = np.array([[-1,0,0,15],[0,-1,0,15],[0,0,1,0],[0,0,0,1]],float)
    source_affine = np.eye(4)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[3,5,6] = 3
    image = nib.Nifti1Image(mask, source_affine)
    aligned = pilot.align_label_volume(image, shape, reference_affine, 'synthetic-mask')
    assert aligned.shape == shape
    assert aligned[12,10,6] == 3
    assert set(np.unique(aligned)) <= {0,1,2,3}
