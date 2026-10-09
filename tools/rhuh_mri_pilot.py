#!/usr/bin/env python3
"""Patient-separated, label-free 3D MRI prediction pilot for MYCELIA.

This is an experimental patch memory, not a clinical diagnostic system. Expert
segmentations are loaded only after the label-free prediction and prototype
learning steps, for descriptive post-hoc alignment metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from mycelia.volumetric_memory import (
    GroupedPredictiveMemory, VolumetricMemoryConfig, VolumetricMycelium,
)

MODALITIES = ("t1", "t1ce", "t2", "flair", "adc")
MIRROR_MODALITIES = {"T1": "t1", "T1C": "t1ce", "T2": "t2", "FLAIR": "flair", "ADC": "adc"}
MIRROR_SOURCE = "AIOmarRehan/medtrace-rhuh-gbm-derived@713d099a2ac6ccd3852f1ad9acdc3b5368c4df3a"
SEG = "segmentations"


def patient_visits(root: Path, modalities=MODALITIES):
    """Find complete studies in the TCIA layout or the indexed public mirror."""
    index = root / "index.csv"
    if index.is_file():
        studies = {}
        with index.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("kind") not in {"images", "masks"}:
                    continue
                modality = MIRROR_MODALITIES.get(row.get("sequence", ""))
                if modality not in modalities:
                    continue
                visit = {"preoperative": "0", "early-postoperative": "1",
                         "recurrence": "2"}.get(row.get("timepoint", ""))
                if visit is None:
                    continue
                key = (row["subject"], visit)
                slot = "images" if row["kind"] == "images" else "masks"
                studies.setdefault(key, {"images": {}, "masks": {}})[slot][modality] = root / row["path"]
        visits = {}
        for (patient, visit), study in sorted(studies.items()):
            if not set(modalities).issubset(study["images"]):
                continue
            # Each image has a mask aligned to its own sequence grid. Use T1C
            # when available; load_visit verifies all selected grids match.
            mask_modality = "t1ce" if "t1ce" in modalities else modalities[0]
            mask = study["masks"].get(mask_modality)
            if mask is None:
                continue
            files = {m: study["images"][m] for m in modalities}
            files[SEG] = mask
            visits.setdefault(patient, []).append((visit, files))
        return visits

    visits = {}
    for patient_dir in sorted(root.glob("RHUH-*")):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        for visit_dir in sorted((p for p in patient_dir.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
            stem = f"{patient}_{visit_dir.name}"
            files = {m: visit_dir / f"{stem}_{m}.nii.gz" for m in modalities}
            files[SEG] = visit_dir / f"{stem}_{SEG}.nii.gz"
            # One source-release filename is an outlier; TCIA's archive table
            # and the published collection manifest identify this same mask.
            if not files[SEG].is_file() and (visit_dir / "segmentation.nii.gz").is_file():
                files[SEG] = visit_dir / "segmentation.nii.gz"
            if all(p.is_file() for p in files.values()):
                visits.setdefault(patient, []).append((visit_dir.name, files))
    return visits


def align_label_volume(mask_img, reference_shape, reference_affine, source_name='<mask>'):
    """Return discrete labels on an MRI reference grid, preserving label IDs."""
    if len(mask_img.shape) != 3:
        raise ValueError(f"expected 3D segmentation: {source_name} has {mask_img.shape}")
    if mask_img.shape != reference_shape or not np.allclose(mask_img.affine, reference_affine, atol=1e-3):
        from nibabel.processing import resample_from_to
        mask_img = resample_from_to(mask_img, (reference_shape, reference_affine),
                                    order=0, mode='constant', cval=0)
    mask_data = np.asarray(mask_img.dataobj)
    mask_codes = np.unique(mask_data)
    if not np.all(np.isin(mask_codes, (0, 1, 2, 3))):
        raise ValueError(f"unexpected segmentation labels in {source_name}: {mask_codes[:20]}")
    return mask_data


def visual_patch_descriptors(volume, patch=4):
    """Fixed, label-free mean/texture/gradient cues for each 3D patch."""
    value = np.asarray(volume, dtype=np.float32)
    if (value.ndim != 3 or min(value.shape) < patch or
            any(size % patch for size in value.shape) or
            not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1))):
        raise ValueError('normalized volume must be finite 3D [0,1] data divisible by patch size')
    grid = tuple(size // patch for size in value.shape)
    blocks = value.reshape(grid[0], patch, grid[1], patch,
                           grid[2], patch).transpose(0, 2, 4, 1, 3, 5)
    mean = blocks.mean(axis=(3, 4, 5))
    texture = blocks.std(axis=(3, 4, 5))
    gradients = np.gradient(value)
    magnitude = np.sqrt(sum(component*component for component in gradients))
    gradient_blocks = magnitude.reshape(
        grid[0], patch, grid[1], patch, grid[2], patch
    ).transpose(0, 2, 4, 1, 3, 5)
    gradient_mean = gradient_blocks.mean(axis=(3, 4, 5))
    return np.stack((mean, np.clip(2.0*texture, 0, 1),
                     np.clip(4.0*gradient_mean, 0, 1)), axis=-1)


_DCT8_FREQUENCIES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                     (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0))
_DCT64_FREQUENCIES = tuple(product(range(4), repeat=3))


def visual_patch_dct(volume, frequencies, patch=4):
    """Signed 3D DCT patch features, normalized by fixed coefficient bounds."""
    value = np.asarray(volume, dtype=np.float32)
    if (value.ndim != 3 or patch != 4 or min(value.shape) < patch or
            any(size % patch for size in value.shape) or
            not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1))):
        raise ValueError('patch DCT expects finite 3D [0,1] data divisible into 4-voxel patches')
    grid = tuple(size // patch for size in value.shape)
    blocks = value.reshape(grid[0], patch, grid[1], patch,
                           grid[2], patch).transpose(0, 2, 4, 1, 3, 5)
    positions = np.arange(patch, dtype=np.float64)
    basis_indices = np.arange(patch, dtype=np.float64)[:, None]
    basis = np.sqrt(2.0 / patch) * np.cos(np.pi*(2*positions+1)*basis_indices/(2*patch))
    basis[0] = 1.0/np.sqrt(patch)
    features = []
    for kz, ky, kx in frequencies:
        kernel = np.einsum('i,j,k->ijk', basis[kz], basis[ky], basis[kx])
        # The L1 norm is an image-independent upper bound for this coefficient.
        bound = float(np.abs(kernel).sum())
        coefficient = np.einsum('...ijk,ijk->...', blocks, kernel, optimize=True)
        features.append(coefficient / bound)
    return np.stack(features, axis=-1).astype(np.float32)


def visual_patch_dct8(volume, patch=4):
    """Eight signed low-frequency DCT cues for compact contextual input."""
    return visual_patch_dct(volume, _DCT8_FREQUENCIES, patch)


def visual_patch_dct64(volume, patch=4):
    """All 64 signed orthonormal frequency modes for 4×4×4 patch content."""
    return visual_patch_dct(volume, _DCT64_FREQUENCIES, patch)


def load_visit(files, factor=4, patch=4, modalities=MODALITIES,
               visual_representation='summary'):
    arrays = {}
    shape = None
    affine = None
    for key in modalities:
        path = files[key]
        img = nib.load(str(path))
        if len(img.shape) != 3:
            raise ValueError(f"expected 3D NIfTI: {path} has {img.shape}")
        if shape is None:
            shape, affine = img.shape, img.affine
        elif img.shape != shape or not np.allclose(img.affine, affine, atol=1e-3):
            raise ValueError(f"unaligned MRI volumes in visit: {path}")
        # Registered archive volumes share a grid; deterministic stride sampling
        # makes this exploratory 3D pilot manageable without an interpolation lib.
        a = np.asanyarray(img.dataobj)[::factor, ::factor, ::factor]
        arrays[key] = np.asarray(a)
    mask_path = files[SEG]
    mask_img = nib.load(str(mask_path))
    # Nearest-neighbor interpolation preserves the four discrete tumor labels.
    mask_data = align_label_volume(mask_img, shape, affine, mask_path)
    arrays[SEG] = mask_data[::factor, ::factor, ::factor]
    common = tuple(min(a.shape[i] for a in arrays.values()) for i in range(3))
    common = tuple((n // patch) * patch for n in common)
    if min(common) < patch:
        raise ValueError(f"volume too small after downsampling: {common}")
    arrays = {k: a[tuple(slice(0, n) for n in common)] for k, a in arrays.items()}
    norm = []
    visual_channels = []
    for modality in modalities:
        a = arrays[modality].astype(np.float32, copy=False)
        foreground = np.isfinite(a) & (a != 0)
        vals = a[foreground]
        if vals.size < 100:
            raise ValueError(f"too few finite nonzero voxels for {modality}")
        lo, hi = np.percentile(vals, [1, 99])
        if not np.isfinite(lo + hi) or hi <= lo:
            raise ValueError(f"invalid intensity range for {modality}")
        a = np.clip((np.nan_to_num(a, nan=0.0, posinf=hi, neginf=lo) - lo) / (hi - lo), 0, 1)
        if visual_representation == 'summary':
            descriptors = visual_patch_descriptors(a, patch)
            norm.append(descriptors[..., 0])
            # Compact fixed intensity, texture, and gradient cues.
            visual_channels.extend((descriptors[..., i] for i in range(3)))
        elif visual_representation in {'dct8', 'dct64'}:
            frequencies = (_DCT8_FREQUENCIES if visual_representation == 'dct8'
                          else _DCT64_FREQUENCIES)
            coefficients = visual_patch_dct(a, frequencies, patch)
            norm.append(coefficients[..., 0])
            visual_channels.extend(coefficients[..., i] for i in range(len(frequencies)))
        else:
            raise ValueError(f'unknown visual representation: {visual_representation}')
    seg = arrays[SEG].astype(np.int16, copy=False)
    labels = seg.reshape(common[0] // patch, patch,
                         common[1] // patch, patch,
                         common[2] // patch, patch).transpose(0, 2, 4, 1, 3, 5)
    labels = labels.reshape(*labels.shape[:3], -1)
    counts = np.stack([(labels == code).sum(axis=-1) for code in range(4)], axis=-1)
    patch_labels = counts.argmax(axis=-1).astype(np.uint8)
    y = np.stack(norm, axis=-1).reshape(-1, len(modalities))
    visual = np.stack(visual_channels, axis=-1).reshape(-1, len(visual_channels))
    lbl = patch_labels.reshape(-1)
    coords = np.indices(patch_labels.shape).reshape(3, -1).T
    # Keep patches with appreciable observed MRI signal; do not use segmentation.
    keep = (y > 0.01).any(axis=1)
    return y[keep], visual[keep], lbl[keep], coords[keep], patch_labels.shape


def context_features(grid):
    """For each patch, use six orthogonal neighbors and return center targets.

    Each output modality's center value is excluded from its predictors. This
    makes the task a real local masked-patch prediction instead of recall.
    """
    channels = grid.shape[-1]
    neighbors = []
    for axis in range(3):
        for delta in (-1, 1):
            shifted = np.zeros_like(grid)
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            if delta == 1:
                dst[axis], src[axis] = slice(0, -1), slice(1, None)
            else:
                dst[axis], src[axis] = slice(1, None), slice(0, -1)
            shifted[tuple(dst)] = grid[tuple(src)]
            neighbors.append(shifted.reshape(-1, channels))
    neighbor_context = np.concatenate(neighbors, axis=1)
    center = grid.reshape(-1, channels)
    pairs = []
    for m in range(channels):
        # Withhold this modality's central patch value from its own predictors.
        pairs.append((np.column_stack((neighbor_context, np.delete(center, m, axis=1))), center[:, m]))
    return pairs


def grid_for_visit(X, coords, shape):
    grid = np.zeros((*shape, X.shape[1]), np.float32)
    grid[tuple(coords.T)] = X
    return grid


def patient_split(patients, seed, fold, folds):
    if not 2 <= folds <= len(patients):
        raise ValueError("folds must be between 2 and the number of patients")
    if not 0 <= fold < folds:
        raise ValueError("fold index is outside the configured range")
    shuffled = np.asarray(sorted(patients), dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    test_ids = set(np.array_split(shuffled, folds)[fold].tolist())
    train_ids = set(patients) - test_ids
    return train_ids, test_ids


def six_neighbor_means(grid, coords, fallback):
    """Mean visible 3D neighbor vectors for the requested patch coordinates."""
    result = np.empty((len(coords), grid.shape[-1]), dtype=float)
    shape = grid.shape[:3]
    offsets = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    for index, coordinate in enumerate(coords):
        values = []
        for offset in offsets:
            neighbor = tuple(coordinate+offset)
            if all(0 <= neighbor[d] < shape[d] for d in range(3)):
                values.append(grid[neighbor])
        result[index] = np.mean(values, axis=0) if values else fallback
    return result


def calibrate_native_route_weight(train_visits, train_patients, args):
    """Select a route/local blend using an inner patient split of outer train."""
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    calibration_train, calibration_patients = patient_split(
        sorted(train_patients), seed=args.seed+50_003+args.fold,
        fold=0, folds=min(args.folds, len(train_patients)))
    training_volumes = [grid_for_visit(X, coords, shape)
                        for patient, visit, X, labels, coords, shape, *_ in train_visits
                        if patient in calibration_train]
    model = VolumetricMycelium(VolumetricMemoryConfig(
        max_colonies=args.clusters, novelty_threshold=0.08,
        learning_rate=0.04, context_bandwidth=0.18))
    if getattr(args, "native_streaming", False):
        for volume in training_volumes:
            model.partial_fit(volume, seed=args.seed+60_003+args.fold)
    else:
        model.fit(training_volumes, seed=args.seed+60_003+args.fold)
    per_patient = {patient: {weight: [] for weight in weights}
                   for patient in calibration_patients}
    for patient, visit, X, labels, coords, shape, *_ in train_visits:
        if patient not in calibration_patients:
            continue
        grid = grid_for_visit(X, coords, shape)
        routes = model.predict_volume(grid)['predictions'][tuple(coords.T)]
        local = six_neighbor_means(grid, coords, X.mean(axis=0))
        for weight in weights:
            estimate = weight*routes+(1-weight)*local
            per_patient[patient][weight].append(np.mean((estimate-X)**2, axis=0))
    patient_macro = {
        weight: float(np.mean([np.mean(np.concatenate(per_patient[p][weight], axis=0))
                               for p in calibration_patients]))
        for weight in weights
    }
    selected = min(weights, key=lambda weight: (patient_macro[weight], weight))
    return selected, {
        'inner_train_patients': sorted(calibration_train),
        'validation_patients': sorted(calibration_patients),
        'patient_macro_mse_by_route_weight': {str(k): v for k, v in patient_macro.items()},
        'selected_route_weight': selected,
    }


def calibrate_local_predictive_route_weight(train_visits, train_patients, args):
    """Tune how much neighboring route votes steer local predictive colonies."""
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    calibration_train, validation_patients = patient_split(
        sorted(train_patients), seed=args.seed+70_003+args.fold,
        fold=0, folds=min(args.folds, len(train_patients)))
    training_volumes = [grid_for_visit(X, coords, shape)
                        for patient, visit, X, labels, coords, shape, *_ in train_visits
                        if patient in calibration_train]
    model = VolumetricMycelium(VolumetricMemoryConfig(
        max_colonies=args.clusters, novelty_threshold=0.08,
        learning_rate=0.04, context_bandwidth=0.18))
    if getattr(args, "native_streaming", False):
        for volume in training_volumes:
            model.partial_fit(volume, seed=args.seed+80_003+args.fold)
    else:
        model.fit(training_volumes, seed=args.seed+80_003+args.fold)
    channels = training_volumes[0].shape[-1]
    patient_errors = {patient: {modality: {weight: [] for weight in weights}
                                for modality in range(channels)}
                      for patient in validation_patients}
    for patient, visit, X, labels, coords, shape, *_ in train_visits:
        if patient not in validation_patients:
            continue
        grid = grid_for_visit(X, coords, shape)
        for weight in weights:
            predicted = model.predict_local_volume(grid, route_weight=weight)["predictions"][tuple(coords.T)]
            for modality in range(channels):
                patient_errors[patient][modality][weight].append(
                    float(np.mean((predicted[:, modality]-X[:, modality])**2)))
    patient_macro = {
        modality: {weight: float(np.mean([
            np.mean(patient_errors[patient][modality][weight])
            for patient in validation_patients
        ])) for weight in weights}
        for modality in range(channels)
    }
    selected = [min(weights, key=lambda weight: (patient_macro[m][weight], weight))
                for m in range(channels)]
    return selected, {
        "inner_train_patients": sorted(calibration_train),
        "validation_patients": sorted(validation_patients),
        "patient_macro_mse_by_route_weight": {
            str(modality): {str(weight): value for weight, value in rows.items()}
            for modality, rows in patient_macro.items()
        },
        "selected_route_weight_by_modality": selected,
    }


def calibrate_grouped_specialist_influence(train_visits, train_patients, args, channel_groups,
                                           *, context_feature_channels=None,
                                           fit_predictors=True, target_memory=None):
    """Calibrate per-colony specialists on inner held-out patients only.

    A colony receives specialist influence only when enough independent patient
    evidence supports it and the specialist improves over the shared predictor.
    """
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    min_patches = 128
    min_patient_patches = 16
    min_patients = 2
    min_relative_improvement = 0.01
    inner_train, validation_patients = patient_split(
        sorted(train_patients), seed=args.seed+90_003+args.fold,
        fold=0, folds=min(args.folds, len(train_patients)))
    training_volumes = [grid_for_visit(record[6], record[4], record[5])
                        for record in train_visits if record[0] in inner_train]
    memory = VolumetricMycelium(VolumetricMemoryConfig(
        max_colonies=args.clusters, novelty_threshold=0.08,
        learning_rate=0.04, context_bandwidth=0.18))
    if getattr(args, 'native_streaming', False) and fit_predictors:
        for volume in training_volumes:
            memory.partial_fit(volume, seed=args.seed+91_003+args.fold)
    else:
        memory.fit(training_volumes, seed=args.seed+91_003+args.fold,
                   fit_predictors=fit_predictors)
    predictor = GroupedPredictiveMemory(memory, channel_groups,
                                        context_scales=(1, 2),
                                        context_feature_channels=context_feature_channels).fit(training_volumes)
    # Errors retain patient boundaries; patches from a single scan cannot
    # masquerade as independent calibration evidence.
    errors = {
        patient: {group: {colony: {weight: [] for weight in weights}
                          for colony in range(args.clusters)}
                  for group in range(len(channel_groups))}
        for patient in validation_patients
    }
    for record in train_visits:
        patient, _, _, _, coords, shape, visual = record[:7]
        if patient not in validation_patients:
            continue
        grid = grid_for_visit(visual, coords, shape)
        result = predictor.predict_volume(grid)
        flat_indices = np.ravel_multi_index(coords.T, shape)
        for group_index, group in enumerate(channel_groups):
            visible = [index for index in range(grid.shape[-1]) if index not in group]
            valid = np.max(np.abs(grid[..., visible]), axis=-1).reshape(-1)[flat_indices] > memory.config.activity_threshold
            if not valid.any():
                continue
            indices = flat_indices[valid]
            actual = grid.reshape(-1, grid.shape[-1])[indices][:, list(group)]
            shared = result['global_predictions'].reshape(-1, grid.shape[-1])[indices][:, list(group)]
            specialist = result['local_predictions'].reshape(-1, grid.shape[-1])[indices][:, list(group)]
            assignments = result['colony_assignments'].reshape(-1, len(channel_groups))[indices, group_index]
            for colony in np.unique(assignments):
                if colony < 0 or colony >= args.clusters:
                    continue
                selected = assignments == colony
                target = actual[selected]
                base = shared[selected]
                local = specialist[selected]
                for weight in weights:
                    prediction = (1.0-weight)*base + weight*local
                    errors[patient][group_index][int(colony)][weight].extend(
                        np.mean((prediction-target)**2, axis=1).tolist())
    selected = np.zeros((len(channel_groups), args.clusters), dtype=float)
    audits = {}
    for group_index in range(len(channel_groups)):
        audits[str(group_index)] = {}
        for colony in range(args.clusters):
            patient_counts = {
                patient: len(errors[patient][group_index][colony][0.0])
                for patient in validation_patients
            }
            available = [patient for patient, count in patient_counts.items() if count]
            total = sum(patient_counts.values())
            qualified = [patient for patient, count in patient_counts.items()
                         if count >= min_patient_patches]
            row = None
            reason = None
            if total < min_patches:
                reason = 'insufficient_total_patches'
            elif len(qualified) < min_patients:
                reason = 'insufficient_independent_patients'
            else:
                # Patient-macro loss gives each patient equal influence.
                row = {weight: float(np.mean([
                    np.mean(errors[patient][group_index][colony][weight])
                    for patient in qualified
                ])) for weight in weights}
                best = min(weights, key=lambda weight: (row[weight], weight))
                baseline = row[0.0]
                improvement = (baseline-row[best])/max(baseline, 1e-12)
                if best == 0.0:
                    reason = 'shared_predictor_wins'
                elif improvement < min_relative_improvement:
                    reason = 'improvement_below_1_percent'
                else:
                    selected[group_index, colony] = best
            audits[str(group_index)][str(colony)] = {
                'patients_with_patches': len(available),
                'patients_meeting_minimum': len(qualified),
                'patches_scored': total,
                'patches_by_patient': patient_counts,
                'patient_macro_mse_by_weight': (
                    {str(weight): value for weight, value in row.items()} if row else None
                ),
                'selected_weight': float(selected[group_index, colony]),
                'fallback_reason': reason,
            }
    alignment = None
    if target_memory is not None:
        selected, alignment = align_colony_specialist_weights(
            selected, memory.centers, target_memory.centers,
            capacity=args.clusters)
    return selected, {
        'inner_train_patients': sorted(inner_train),
        'validation_patients': sorted(validation_patients),
        'colony_calibration_by_group': audits,
        'selected_local_weight_by_group_and_colony': selected.tolist(),
        'inner_to_outer_colony_alignment': alignment,
        'minimum_total_patches': min_patches,
        'minimum_patches_per_patient': min_patient_patches,
        'minimum_patients': min_patients,
        'minimum_relative_improvement': min_relative_improvement,
        'context_scales': [1, 2],
        'context_feature_channels': (list(context_feature_channels)
                                     if context_feature_channels is not None else None),
        'target_group_activity_uses_visible_channels_only': True,
        'labels_used_for_calibration': False,
    }


def align_colony_specialist_weights(weights, source_centers, target_centers, *, capacity=None):
    """Transfer inner-split specialist weights by matching visual prototypes.

    Colony indices are fit-order artifacts. A one-to-one minimum-distance
    assignment aligns the independently fit inner and outer banks before
    applying weights. Unmatched target colonies safely retain weight zero.
    """
    weights = np.asarray(weights, dtype=float)
    source = np.asarray(source_centers, dtype=float)
    target = np.asarray(target_centers, dtype=float)
    if (weights.ndim != 2 or source.ndim != 2 or target.ndim != 2 or
            source.shape[1:] != target.shape[1:] or weights.shape[1] < len(source) or
            not len(source) or not len(target) or not np.all(np.isfinite(source)) or
            not np.all(np.isfinite(target))):
        raise ValueError('colony weights and prototype banks must be finite and dimensionally aligned')
    squared = np.mean((source[:, None, :]-target[None, :, :])**2, axis=-1)
    source_ids, target_ids = linear_sum_assignment(squared)
    output_columns = max(len(target), int(capacity or len(target)))
    aligned = np.zeros((weights.shape[0], output_columns), dtype=float)
    aligned[:, target_ids] = weights[:, source_ids]
    matches = [{
        'inner_colony': int(source_id), 'outer_colony': int(target_id),
        'prototype_rms_distance': float(np.sqrt(squared[source_id, target_id])),
        'transferred_group_weights': weights[:, source_id].tolist(),
    } for source_id, target_id in zip(source_ids, target_ids)]
    return aligned, {
        'method': 'Hungarian one-to-one minimum RMS distance on learned visual prototypes',
        'inner_colonies': len(source), 'outer_colonies': len(target),
        'matches': matches,
        'unmatched_inner_colonies': sorted(set(range(len(source)))-set(source_ids.tolist())),
        'unmatched_outer_colonies': sorted(set(range(len(target)))-set(target_ids.tolist())),
        'unmatched_outer_weight': 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--fold", type=int, default=0, help="patient-level held-out fold index")
    ap.add_argument("--folds", type=int, default=5, help="number of disjoint patient-level folds")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--visual-representation", choices=("summary", "dct8"), default="summary",
                    help="summary uses mean/texture/gradient; dct8 retains signed low-frequency 3D patch structure")
    ap.add_argument("--native-streaming", action="store_true",
                    help="feed training volumes sequentially through VolumetricMycelium.partial_fit")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--max-patients", type=int, default=0, help="0 means all complete patients")
    ap.add_argument("--modalities", default="auto",
                    help="comma-separated sequences, or auto for those indexed in the dataset")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.modalities == "auto":
        if (args.dataset / "index.csv").is_file():
            with (args.dataset / "index.csv").open(newline="") as stream:
                available = {MIRROR_MODALITIES[row.get("sequence", "")]
                             for row in csv.DictReader(stream)
                             if row.get("kind") == "images" and
                             row.get("sequence", "") in MIRROR_MODALITIES}
            modalities = tuple(m for m in MODALITIES if m in available)
        else:
            modalities = MODALITIES
    else:
        requested = tuple(m.strip().lower() for m in args.modalities.split(",") if m.strip())
        if not requested or len(set(requested)) != len(requested) or any(m not in MODALITIES for m in requested):
            ap.error("--modalities must be 'auto' or a comma-separated subset of " + ",".join(MODALITIES))
        modalities = requested
    if len(modalities) < 2:
        raise SystemExit(f"need at least two MRI modalities; found {modalities}")
    visits = patient_visits(args.dataset, modalities)
    patients = sorted(visits)
    if args.max_patients:
        patients = patients[:args.max_patients]
    if len(patients) < 5:
        raise SystemExit(f"need at least five complete patients; found {len(patients)}")
    train_ids, test_ids = patient_split(patients, args.seed, args.fold, args.folds)
    assert not train_ids.intersection(test_ids)

    data = {"train": [], "test": []}
    per_visit_audit = []
    for split, ids in (("train", sorted(train_ids)), ("test", sorted(test_ids))):
        for patient in ids:
            for visit, files in visits[patient]:
                X, visual, labels, coords, shape = load_visit(
                    files, args.factor, args.patch, modalities, args.visual_representation)
                per_visit_audit.append({"patient": patient, "visit": visit, "split": split,
                                        "patch_grid": list(shape), "patches": int(len(X)),
                                        "mask_labels": sorted(np.unique(labels).tolist())})
                data[split].append((patient, visit, X, labels, coords, shape, visual))

    train_x = np.concatenate([v[2] for v in data["train"]])
    test_x = np.concatenate([v[2] for v in data["test"]])
    train_fingerprints = {
        hashlib.sha256((str(v[5]).encode() + v[2].astype("<f4", copy=False).tobytes())).hexdigest()
        for v in data["train"]
    }
    test_fingerprints = {
        hashlib.sha256((str(v[5]).encode() + v[2].astype("<f4", copy=False).tobytes())).hexdigest()
        for v in data["test"]
    }
    cross_split_duplicates = sorted(train_fingerprints & test_fingerprints)
    if cross_split_duplicates:
        raise SystemExit(f"found duplicate downsampled MRI inputs across patient split: {cross_split_duplicates}")
    scaler = StandardScaler().fit(train_x)
    train_z, test_z = scaler.transform(train_x), scaler.transform(test_x)
    k = min(args.clusters, len(train_z))
    km = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=8192,
                         n_init=3, max_iter=100, reassignment_ratio=0.01)
    km.fit(train_z)
    train_labels = np.concatenate([v[3] for v in data["train"]])
    test_labels = np.concatenate([v[3] for v in data["test"]])
    train_cluster = km.predict(train_z)
    test_cluster = km.predict(test_z)
    train_visual = np.concatenate([v[6] for v in data["train"]])
    test_visual = np.concatenate([v[6] for v in data["test"]])
    visual_scaler = StandardScaler().fit(train_visual)
    train_visual_z = visual_scaler.transform(train_visual)
    test_visual_z = visual_scaler.transform(test_visual)
    visual_km = MiniBatchKMeans(n_clusters=k, random_state=args.seed+1, batch_size=8192,
                                n_init=3, max_iter=100, reassignment_ratio=0.01)
    visual_km.fit(train_visual_z)
    test_visual_clusters = visual_km.predict(test_visual_z)
    visual_nonbackground = test_labels > 0
    visual_cluster_alignment = None
    if visual_nonbackground.any():
        visual_cluster_alignment = {
            "n_nonbackground_patches": int(visual_nonbackground.sum()),
            "adjusted_rand_index": float(adjusted_rand_score(
                test_labels[visual_nonbackground], test_visual_clusters[visual_nonbackground])),
            "normalized_mutual_information": float(normalized_mutual_info_score(
                test_labels[visual_nonbackground], test_visual_clusters[visual_nonbackground])),
        }
    visual_memory = VolumetricMycelium(VolumetricMemoryConfig(
        max_colonies=args.clusters, novelty_threshold=0.08,
        learning_rate=0.04, context_bandwidth=0.18))
    visual_training_grids = [grid_for_visit(v[6], v[4], v[5]) for v in data["train"]]
    if getattr(args, "native_streaming", False):
        for visual_grid in visual_training_grids:
            visual_memory.partial_fit(visual_grid, seed=args.seed+1)
    else:
        visual_memory.fit(visual_training_grids, seed=args.seed+1)
    visual_memory_state = visual_memory.validate()
    features_per_modality = 3 if args.visual_representation == 'summary' else len(_DCT8_FREQUENCIES)
    visual_feature_names = (('patch_mean', 'patch_std', 'gradient_magnitude')
                            if args.visual_representation == 'summary' else
                            tuple(f'dct_{z}{y}{x}' for z, y, x in _DCT8_FREQUENCIES))
    visual_channel_groups = [tuple(range(features_per_modality*m,
                                         features_per_modality*(m+1)))
                             for m in range(len(modalities))]
    grouped_specialist_weights, grouped_specialist_calibration = calibrate_grouped_specialist_influence(
        data['train'], train_ids, args, visual_channel_groups,
        target_memory=visual_memory)
    grouped_memory = GroupedPredictiveMemory(
        visual_memory, visual_channel_groups, context_scales=(1, 2)).fit(visual_training_grids)
    grouped_ridge = []
    for group_index in range(len(visual_channel_groups)):
        examples = [grouped_memory.masked_examples(grid, group_index)[:2]
                    for grid in visual_training_grids]
        features = np.concatenate([item[0] for item in examples])
        targets = np.concatenate([item[1] for item in examples])
        grouped_ridge.append(Ridge(alpha=10.0).fit(features, targets))
    grouped_prediction_metrics = {
        modality: {"n": 0, "model_squared_error": np.zeros(features_per_modality),
                   "ridge_squared_error": np.zeros(features_per_modality)}
        for modality in modalities
    }
    visual_colony_clusters = []
    for visit in data["test"]:
        visual_grid = grid_for_visit(visit[6], visit[4], visit[5])
        visual_assignments = visual_memory.assignments(visual_grid)
        visual_colony_clusters.append(visual_assignments[tuple(visit[4].T)])
        grouped_prediction = grouped_memory.predict_volume(
            visual_grid, local_weights=grouped_specialist_weights)
        grouped_predictions = grouped_prediction["predictions"]
        flat_indices = np.ravel_multi_index(visit[4].T, visit[5])
        actual = visit[6]
        for modality_index, modality in enumerate(modalities):
            group = visual_channel_groups[modality_index]
            visible = [index for index in range(visual_grid.shape[-1]) if index not in group]
            valid = np.max(np.abs(visual_grid.reshape(-1, visual_grid.shape[-1])[flat_indices][:, visible]), axis=1) > visual_memory.config.activity_threshold
            if not valid.any():
                continue
            model_prediction = grouped_predictions[tuple(visit[4].T)][valid][:, list(group)]
            target = actual[valid][:, list(group)]
            context = grouped_memory.context_matrix(visual_grid, group, (1, 2))[flat_indices[valid]]
            ridge_prediction = grouped_ridge[modality_index].predict(context)
            residual = model_prediction-target
            ridge_residual = ridge_prediction-target
            scores = grouped_prediction_metrics[modality]
            scores["n"] += int(valid.sum())
            scores["model_squared_error"] += np.sum(residual*residual, axis=0)
            scores["ridge_squared_error"] += np.sum(ridge_residual*ridge_residual, axis=0)
    visual_colony_clusters = np.concatenate(visual_colony_clusters)
    grouped_prediction_results = {}
    descriptor_names = visual_feature_names
    for modality in modalities:
        scores = grouped_prediction_metrics[modality]
        grouped_prediction_results[modality] = {
            "n_patches": scores["n"],
            "features": {
                name: {
                    "mycelia_mse": float(scores["model_squared_error"][i]/max(1, scores["n"])),
                    "mycelia_rmse": float(np.sqrt(scores["model_squared_error"][i]/max(1, scores["n"]))),
                    "ridge_mse": float(scores["ridge_squared_error"][i]/max(1, scores["n"])),
                    "ridge_rmse": float(np.sqrt(scores["ridge_squared_error"][i]/max(1, scores["n"]))),
                } for i, name in enumerate(descriptor_names)
            },
            "selected_local_specialist_weight": float(np.mean(
                grouped_specialist_weights[modalities.index(modality)])),
            "specialized_colony_count": int(np.count_nonzero(
                grouped_specialist_weights[modalities.index(modality)])),
        }
    colony_visual_alignment = None
    if visual_nonbackground.any():
        colony_visual_alignment = {
            "n_nonbackground_patches": int(visual_nonbackground.sum()),
            "adjusted_rand_index": float(adjusted_rand_score(
                test_labels[visual_nonbackground], visual_colony_clusters[visual_nonbackground])),
            "normalized_mutual_information": float(normalized_mutual_info_score(
                test_labels[visual_nonbackground], visual_colony_clusters[visual_nonbackground])),
        }
    # Fit local predictive weights on training patients only. All timepoints
    # from a patient stay in the same partition.
    train_context = [[] for _ in modalities]
    for patient, visit, X, labels, coords, shape, *_ in data["train"]:
        grid = grid_for_visit(X, coords, shape)
        pairs = context_features(grid)
        keep = (grid > .01).any(axis=-1).reshape(-1)
        for m in range(len(modalities)):
            tx, ty = pairs[m]
            train_context[m].append((tx[keep], ty[keep]))
    predictors = []
    for m in range(len(modalities)):
        tx = np.concatenate([p[0] for p in train_context[m]])
        ty = np.concatenate([p[1] for p in train_context[m]])
        predictors.append(Ridge(alpha=10.0).fit(tx, ty))

    # Native label-free MYCELIA candidate: online patch colonies compete for a
    # finite memory, then reinforce six-direction routes from training volumes.
    native_memory = VolumetricMycelium(VolumetricMemoryConfig(
        max_colonies=args.clusters, novelty_threshold=0.08,
        learning_rate=0.04, context_bandwidth=0.18))
    native_grids = [grid_for_visit(X, coords, shape)
                    for patient, visit, X, labels, coords, shape, *_ in data["train"]]
    if getattr(args, "native_streaming", False):
        for grid in native_grids:
            native_memory.partial_fit(grid, seed=args.seed)
        native_state = native_memory.validate()
    else:
        native_memory.fit(native_grids, seed=args.seed)
        native_state = native_memory.validate()
    native_memory_path = args.output / "native-volumetric-memory.npz"
    native_memory.save(native_memory_path)
    native_route_weight, native_calibration = calibrate_native_route_weight(
        data["train"], train_ids, args)
    native_predictive_route_weight, native_predictive_calibration = calibrate_local_predictive_route_weight(
        data["train"], train_ids, args)
    if set(native_calibration['validation_patients']) & test_ids:
        raise AssertionError('outer test patient entered native blend calibration')
    if set(native_predictive_calibration['validation_patients']) & test_ids:
        raise AssertionError('outer test patient entered predictive route calibration')

    # Primary: local, cross-modal predictive accuracy against a non-learning
    # neighbor-average baseline, measured only on held-out patients.
    prediction = {m: {"n": 0, "model_squared_error": 0., "neighbor_squared_error": 0.,
                      "global_squared_error": 0.} for m in modalities}
    native_prediction = {m: {"n": 0, "squared_error": 0., "context_available": 0}
                         for m in modalities}
    native_local_prediction = {m: {"n": 0, "squared_error": 0., "route_sources": 0}
                               for m in modalities}
    native_blend_prediction = {m: {"n": 0, "squared_error": 0., "context_available": 0}
                               for m in modalities}
    for patient, visit, X, labels, coords, shape, *_ in data["test"]:
        grid = grid_for_visit(X, coords, shape)
        native = native_memory.predict_volume(grid)
        native_predictions = native["predictions"][tuple(coords.T)]
        native_context = native["context_sources"][tuple(coords.T)]
        native_local_output = native_memory.predict_local_volume(
            grid, route_weight=native_predictive_route_weight)
        native_local = native_local_output["predictions"][tuple(coords.T)]
        native_local_route_context = native_local_output["route_context_sources"][tuple(coords.T)]
        all_coords = np.indices(shape).reshape(3, -1).T
        keep = (grid > .01).any(axis=-1).reshape(-1)
        pairs = context_features(grid)
        keep_indices = np.flatnonzero(keep)
        for m, modality in enumerate(modalities):
            all_input, all_target = pairs[m]
            tx = all_input[keep_indices]
            ty = all_target[keep_indices]
            pred = predictors[m].predict(tx)
            # Neighbor-only comparison is estimated from current patient's
            # adjacent 3D patch values; predictor uses other modalities too.
            target_coords = all_coords[keep_indices]
            neighbor_mean = six_neighbor_means(grid, target_coords, np.mean(ty, axis=0))[:, m]
            global_mean = float(np.mean(train_x[:, m]))
            out = prediction[modality]
            out["n"] += len(ty)
            out["model_squared_error"] += float(np.sum((pred - ty) ** 2))
            out["neighbor_squared_error"] += float(np.sum((neighbor_mean - ty) ** 2))
            out["global_squared_error"] += float(np.sum((global_mean - ty) ** 2))
            native_out = native_prediction[modality]
            native_out["n"] += len(ty)
            native_out["squared_error"] += float(np.sum((native_predictions[:, m]-ty) ** 2))
            native_out["context_available"] += int(np.sum(native_context[:, m] > 0))
            local_out = native_local_prediction[modality]
            local_out["n"] += len(ty)
            local_out["squared_error"] += float(np.sum((native_local[:, m]-ty) ** 2))
            local_out["route_sources"] += int(np.sum(native_local_route_context[:, m]))
            blended = native_route_weight*native_predictions[:, m] + \
                      (1-native_route_weight)*neighbor_mean
            blend_out = native_blend_prediction[modality]
            blend_out["n"] += len(ty)
            blend_out["squared_error"] += float(np.sum((blended-ty) ** 2))
            blend_out["context_available"] += int(np.sum(native_context[:, m] > 0))

    prediction_results = {}
    for m, values in prediction.items():
        n = max(1, values.pop("n"))
        prediction_results[m] = {k: v / n for k, v in values.items()}
        prediction_results[m]["rmse"] = float(np.sqrt(prediction_results[m]["model_squared_error"]))
        prediction_results[m]["neighbor_rmse"] = float(np.sqrt(prediction_results[m]["neighbor_squared_error"]))
        prediction_results[m]["global_mean_rmse"] = float(np.sqrt(prediction_results[m]["global_squared_error"]))

    native_prediction_results = {}
    native_local_results = {}
    native_blend_results = {}
    for modality_index, modality in enumerate(modalities):
        values = native_prediction[modality]
        n = max(1, values["n"])
        mse = values["squared_error"]/n
        native_prediction_results[modality] = {
            "mse": mse, "rmse": float(np.sqrt(mse)), "n_patches": n,
            "route_context_coverage": float(values["context_available"]/n),
        }
        local = native_local_prediction[modality]
        local_mse = local["squared_error"] / max(1, local["n"])
        native_local_results[modality] = {
            "mse": local_mse, "rmse": float(np.sqrt(local_mse)), "n_patches": local["n"],
            "route_context_source_mean": local["route_sources"] / max(1, local["n"]),
            "selected_route_weight": native_predictive_route_weight[modality_index],
        }
        blend = native_blend_prediction[modality]
        blend_mse = blend["squared_error"]/max(1, blend["n"])
        native_blend_results[modality] = {
            "mse": blend_mse, "rmse": float(np.sqrt(blend_mse)),
            "n_patches": blend["n"],
            "route_context_coverage": float(blend["context_available"]/max(1, blend["n"])),
            "selected_route_weight": native_route_weight,
        }

    nonbg = test_labels > 0
    cluster_alignment = None
    if nonbg.any():
        cluster_alignment = {
            "n_nonbackground_patches": int(nonbg.sum()),
            "adjusted_rand_index": float(adjusted_rand_score(test_labels[nonbg], test_cluster[nonbg])),
            "normalized_mutual_information": float(normalized_mutual_info_score(test_labels[nonbg], test_cluster[nonbg]))
        }

    # Transparent post-learning readout: each frozen unsupervised prototype is
    # mapped to its most frequent training-mask label. Test masks are untouched
    # until this point. This evaluates localization of subregions, not diagnosis.
    cluster_label_counts = np.zeros((k, 4), dtype=np.int64)
    np.add.at(cluster_label_counts, (train_cluster, train_labels), 1)
    cluster_to_mask = cluster_label_counts.argmax(axis=1)
    readout_pred = cluster_to_mask[test_cluster]
    cm = confusion_matrix(test_labels, readout_pred, labels=[0, 1, 2, 3])
    class_dice = {}
    for code, name in enumerate(("background", "necrosis", "peritumoral", "enhancing")):
        tp = int(cm[code, code])
        denom = int(2*tp + cm[:, code].sum()-tp + cm[code, :].sum()-tp)
        class_dice[name] = float(2*tp/denom) if denom else None
    tumor_truth, tumor_pred = test_labels > 0, readout_pred > 0
    whole_tumor_dice = float(2*np.sum(tumor_truth & tumor_pred) /
                             max(1, np.sum(tumor_truth)+np.sum(tumor_pred)))
    patient_readout = []
    offset = 0
    for patient, visit, X, labels, coords, shape, *_ in data["test"]:
        n = len(labels)
        patient_truth = labels > 0
        patient_pred = readout_pred[offset:offset+n] > 0
        patient_readout.append({"patient": patient, "visit": visit,
                                "whole_tumor_patch_dice": float(2*np.sum(patient_truth & patient_pred) /
                                    max(1, np.sum(patient_truth)+np.sum(patient_pred)))})
        offset += n
    baseline_cm = confusion_matrix(test_labels, np.zeros_like(test_labels), labels=[0, 1, 2, 3])
    baseline_tumor_dice = float(0.0 if np.any(test_labels > 0) else 1.0)
    readout_results = {
        "method": "frozen 16-cluster prototypes mapped to each cluster's most frequent training-mask class",
        "labels_used_only_after_unsupervised_learning": True,
        "heldout_patch_confusion_matrix_rows_true_cols_predicted": cm.tolist(),
        "heldout_class_dice": class_dice,
        "heldout_whole_tumor_patch_dice": whole_tumor_dice,
        "all_background_baseline_whole_tumor_patch_dice": baseline_tumor_dice,
        "all_background_baseline_confusion_matrix": baseline_cm.tolist(),
        "heldout_patient_visit_whole_tumor_patch_dice": patient_readout,
        "cluster_to_mask_label": {str(i): int(label) for i, label in enumerate(cluster_to_mask)},
        "training_patch_counts_by_label": np.bincount(train_labels, minlength=4).tolist(),
        "heldout_patch_counts_by_label": np.bincount(test_labels, minlength=4).tolist(),
    }

    # A small linear readout uses distances to the frozen unlabeled prototypes.
    # This is the explicit supervised bridge from discovered patterns to mask
    # semantics; its test masks are never used in fitting or threshold choice.
    train_distances, test_distances = km.transform(train_z), km.transform(test_z)
    distance_scaler = StandardScaler().fit(train_distances)
    train_distances = distance_scaler.transform(train_distances)
    test_distances = distance_scaler.transform(test_distances)
    linear_readout = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2500,
                                        random_state=args.seed)
    linear_readout.fit(train_distances, train_labels)
    linear_pred = linear_readout.predict(test_distances)
    linear_cm = confusion_matrix(test_labels, linear_pred, labels=[0, 1, 2, 3])
    linear_dice = {}
    for code, name in enumerate(("background", "necrosis", "peritumoral", "enhancing")):
        tp = int(linear_cm[code, code])
        denom = int(2*tp + linear_cm[:, code].sum()-tp + linear_cm[code, :].sum()-tp)
        linear_dice[name] = float(2*tp/denom) if denom else None
    linear_tumor_truth, linear_tumor_pred = test_labels > 0, linear_pred > 0
    linear_patient_scores = []
    offset = 0
    for patient, visit, X, labels, coords, shape, *_ in data["test"]:
        n = len(labels)
        pt, pp = labels > 0, linear_pred[offset:offset+n] > 0
        linear_patient_scores.append({"patient": patient, "visit": visit,
            "whole_tumor_patch_dice": float(2*np.sum(pt & pp) / max(1, np.sum(pt)+np.sum(pp)))})
        offset += n
    supervised_readout_results = {
        "method": "class-balanced multinomial logistic readout on distances to frozen unlabeled prototypes",
        "labels_used_only_after_unsupervised_learning": True,
        "heldout_patch_confusion_matrix_rows_true_cols_predicted": linear_cm.tolist(),
        "heldout_class_dice": linear_dice,
        "heldout_whole_tumor_patch_dice": float(2*np.sum(linear_tumor_truth & linear_tumor_pred) /
                                                   max(1,np.sum(linear_tumor_truth)+np.sum(linear_tumor_pred))),
        "all_background_baseline_whole_tumor_patch_dice": baseline_tumor_dice,
        "heldout_patient_visit_whole_tumor_patch_dice": linear_patient_scores,
    }

    # Direct intensity-only model is an attribution control: it reveals whether
    # the prototype representation adds value over a plain transparent readout.
    direct_readout = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000,
                                        random_state=args.seed)
    direct_readout.fit(train_z, train_labels)
    direct_pred = direct_readout.predict(test_z)
    direct_cm = confusion_matrix(test_labels, direct_pred, labels=[0, 1, 2, 3])
    direct_dice = {}
    for code, name in enumerate(("background", "necrosis", "peritumoral", "enhancing")):
        tp = int(direct_cm[code, code])
        denom = int(2*tp + direct_cm[:, code].sum()-tp + direct_cm[code, :].sum()-tp)
        direct_dice[name] = float(2*tp/denom) if denom else None
    direct_tumor_truth, direct_tumor_pred = test_labels > 0, direct_pred > 0
    direct_readout_results = {
        "method": "class-balanced multinomial logistic model on five normalized modality patch means (attribution control)",
        "labels_used_only_after_unsupervised_learning": True,
        "heldout_patch_confusion_matrix_rows_true_cols_predicted": direct_cm.tolist(),
        "heldout_class_dice": direct_dice,
        "heldout_whole_tumor_patch_dice": float(2*np.sum(direct_tumor_truth & direct_tumor_pred) /
                                                   max(1,np.sum(direct_tumor_truth)+np.sum(direct_tumor_pred))),
        "all_background_baseline_whole_tumor_patch_dice": baseline_tumor_dice,
    }

    summary = {
        "protocol": "3D, multimodal, patient-level split; no labels enter normalization, scaling, clustering, or predictive learning",
        "task": "masked-center patch prediction from six spatial neighbors plus the other center modalities; unlabeled prototype clustering",
        "learner": "native label-free volumetric colony/routes were trained; Ridge, MiniBatchKMeans, and logistic regression are statistical comparison/readout baselines",
        "dataset_version": (f"TCIA RHUH-GBM CC BY 4.0 derivative; pinned {MIRROR_SOURCE}"
                            if (args.dataset / "index.csv").is_file()
                            else "TCIA RHUH-GBM v1 NIfTI release"),
        "limitations": [
            f"single-center glioblastoma cohort, {len(patients)} patients" +
            ("; this pinned derivative contains 28 of the source cohort's 40 patients" if
             (args.dataset / "index.csv").is_file() else ""),
            "the derivative uses four modalities (T1, T1C, T2, FLAIR); it omits ADC",
            "not tumor-vs-no-tumor or tumor-type diagnosis; all source patients have glioblastoma",
            "coarse exploratory downsampling and patch averages; segmentation metrics are patch-level, not voxel-level",
            "file hashes were checked against the pinned mirror revision, not independently against TCIA release checksums",
            "Ridge/KMeans/logistic outputs are conventional statistical baselines; native colony metrics are reported separately",
            "results are a prototype evaluation, not medical validation",
        ],
        "seed": args.seed, "downsample_factor": args.factor, "patch_voxels_after_downsampling": args.patch,
        "patient_fold": args.fold, "patient_folds": args.folds,
        "train_patient_count": len(train_ids), "test_patient_count": len(test_ids),
        "train_patients": sorted(train_ids), "test_patients": sorted(test_ids),
        "train_visits": len(data["train"]), "test_visits": len(data["test"]),
        "duplicate_audit": {"train_test_patient_intersection": sorted(train_ids & test_ids),
                            "exact_downsampled_input_fingerprints_cross_split": len(cross_split_duplicates)},
        "modalities": list(modalities), "clusters": k,
        "heldout_prediction": prediction_results,
        "native_volumetric_mycelia": {
            "method": "online competitive multimodal patch colonies; local Hebbian-style context-target predictive maps; six-direction route reinforcement; no labels or masks enter fit",
            "training_protocol": "sequential volume-by-volume partial_fit" if args.native_streaming else "single fit over training volumes",
            "training_state": native_state,
            "checkpoint": native_memory_path.name,
            "heldout_prediction": native_prediction_results,
            "local_predictive_heldout_prediction": native_local_results,
            "train_only_calibration": native_calibration,
            "predictive_route_calibration": native_predictive_calibration,
            "route_local_blend_heldout_prediction": native_blend_results,
        },
        "heldout_cluster_alignment_posthoc_only": cluster_alignment,
        "heldout_multiscale_visual_cluster_alignment_posthoc_only": visual_cluster_alignment,
        "heldout_mycelia_visual_colony_alignment_posthoc_only": colony_visual_alignment,
        "mycelia_visual_colony_state": visual_memory_state,
        "whole_modality_group_masked_prediction": grouped_prediction_results,
        "grouped_specialist_influence_calibration": grouped_specialist_calibration,
        "grouped_predictor_context_scales": [1, 2],
        "visual_representation": args.visual_representation,
        "visual_descriptor_features": [f"{modality}_{feature}" for modality in modalities
                                       for feature in visual_feature_names],
        "posthoc_mask_readout": readout_results,
        "linear_mask_readout": supervised_readout_results,
        "direct_intensity_readout_control": direct_readout_results,
        "visits": per_visit_audit,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    model_arrays = {
        "feature_mean": scaler.mean_, "feature_scale": scaler.scale_,
        "prototype_centers": km.cluster_centers_, "cluster_to_mask_label": cluster_to_mask,
        "prototype_distance_mean": distance_scaler.mean_, "prototype_distance_scale": distance_scaler.scale_,
        "linear_readout_coef": linear_readout.coef_, "linear_readout_intercept": linear_readout.intercept_,
        "linear_readout_classes": linear_readout.classes_,
        "direct_readout_coef": direct_readout.coef_, "direct_readout_intercept": direct_readout.intercept_,
        "train_patient_ids": np.asarray(sorted(train_ids)),
        "test_patient_ids": np.asarray(sorted(test_ids)),
    }
    for i, modality in enumerate(modalities):
        model_arrays[f"predictor_{modality}_coef"] = predictors[i].coef_
        model_arrays[f"predictor_{modality}_intercept"] = np.asarray(predictors[i].intercept_)
    np.savez_compressed(args.output / "model.npz", **model_arrays)
    print(json.dumps({k: summary[k] for k in ("train_patient_count", "test_patient_count", "train_visits", "test_visits", "heldout_prediction", "native_volumetric_mycelia", "heldout_cluster_alignment_posthoc_only", "heldout_multiscale_visual_cluster_alignment_posthoc_only", "heldout_mycelia_visual_colony_alignment_posthoc_only", "whole_modality_group_masked_prediction")}, indent=2))


if __name__ == "__main__":
    main()
