#!/usr/bin/env python3
"""Aggregate disjoint patient-fold runs from rhuh_mri_pilot.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_DCT8_FREQUENCIES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                     (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0))


def dct8_projection_mse(feature_mses, patch=4):
    """MSE per source voxel for inverse-DCT patches using only the modeled modes."""
    basis = np.sqrt(2.0/patch)*np.cos(
        np.pi*(2*np.arange(patch)+1)*np.arange(patch)[:, None]/(2*patch))
    basis[0] = 1.0/np.sqrt(patch)
    weighted_squared_error = 0.0
    for frequency, mse in zip(_DCT8_FREQUENCIES, feature_mses):
        kernel = np.einsum('i,j,k->ijk', *(basis[index] for index in frequency))
        coefficient_bound = float(np.abs(kernel).sum())
        weighted_squared_error += float(mse)*coefficient_bound**2
    return weighted_squared_error/(patch**3)


def dice_from_cm(cm):
    names = ("background", "necrosis", "peritumoral", "enhancing")
    result = {}
    for i, name in enumerate(names):
        tp = int(cm[i, i])
        denom = int(2 * tp + cm[:, i].sum() - tp + cm[i, :].sum() - tp)
        result[name] = float(2 * tp / denom) if denom else None
    tumor_tp = int(cm[1:, 1:].sum())
    tumor_false_positive = int(cm[0, 1:].sum())
    tumor_false_negative = int(cm[1:, 0].sum())
    denominator = 2 * tumor_tp + tumor_false_positive + tumor_false_negative
    result["whole_tumor"] = float(2 * tumor_tp / denominator) if denominator else None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    runs = []
    for fold in range(args.folds):
        path = args.fold_root / f"fold-{fold}" / "summary.json"
        if not path.is_file():
            raise SystemExit(f"missing completed fold summary: {path}")
        runs.append(json.loads(path.read_text()))
    test_patients = [p for run in runs for p in run["test_patients"]]
    if len(test_patients) != len(set(test_patients)):
        raise SystemExit("patient appears in more than one held-out fold")
    train_test_overlap = any(set(r["train_patients"]) & set(r["test_patients"]) for r in runs)
    if train_test_overlap:
        raise SystemExit("a fold has patient overlap between train and test")
    total_test_visits = sum(r["test_visits"] for r in runs)
    patch_counts = [sum(v["patches"] for v in r["visits"] if v["split"] == "test") for r in runs]
    if not all(r["duplicate_audit"]["exact_downsampled_input_fingerprints_cross_split"] == 0 for r in runs):
        raise SystemExit("a fold has encoded duplicate inputs crossing its patient split")

    modalities = runs[0]["modalities"]
    prediction = {}
    for modality in modalities:
        result = {}
        n = sum(patch_counts)
        for key in ("model_squared_error", "neighbor_squared_error", "global_squared_error"):
            mse = sum(r["heldout_prediction"][modality][key] * count
                      for r, count in zip(runs, patch_counts)) / max(1, n)
            result[key] = mse
            result[key.replace("squared_error", "rmse")] = float(np.sqrt(mse))
        result["n_patches"] = n
        prediction[modality] = result

    native_prediction = {}
    native_local = {}
    native_blend = {}
    for modality in modalities:
        rows = [run["native_volumetric_mycelia"]["heldout_prediction"][modality] for run in runs]
        n = sum(row["n_patches"] for row in rows)
        mse = sum(row["mse"]*row["n_patches"] for row in rows)/max(1,n)
        native_prediction[modality] = {
            "mse": mse, "rmse": float(np.sqrt(mse)), "n_patches": n,
            "route_context_coverage": sum(row["route_context_coverage"]*row["n_patches"]
                                           for row in rows)/max(1,n),
        }
        local_rows = [run["native_volumetric_mycelia"]["local_predictive_heldout_prediction"][modality]
                      for run in runs]
        local_n = sum(row["n_patches"] for row in local_rows)
        local_mse = sum(row["mse"]*row["n_patches"] for row in local_rows)/max(1,local_n)
        native_local[modality] = {"mse": local_mse, "rmse": float(np.sqrt(local_mse)),
                                  "n_patches": local_n,
                                  "route_context_source_mean": sum(row.get("route_context_source_mean", 0.0)*row["n_patches"]
                                                                   for row in local_rows)/max(1,local_n),
                                  "selected_route_weight_by_fold": [row.get("selected_route_weight", 0.0)
                                                                    for row in local_rows]}
        blend_rows = [run["native_volumetric_mycelia"]["route_local_blend_heldout_prediction"][modality]
                      for run in runs]
        blend_n = sum(row["n_patches"] for row in blend_rows)
        blend_mse = sum(row["mse"]*row["n_patches"] for row in blend_rows)/max(1,blend_n)
        native_blend[modality] = {
            "mse": blend_mse, "rmse": float(np.sqrt(blend_mse)), "n_patches": blend_n,
            "route_context_coverage": sum(row["route_context_coverage"]*row["n_patches"]
                                           for row in blend_rows)/max(1,blend_n),
            "selected_route_weight_by_fold": [row["selected_route_weight"] for row in blend_rows],
        }

    def sum_confusion(section, field):
        matrices = [np.asarray(r[section][field], dtype=np.int64) for r in runs]
        return np.sum(matrices, axis=0)

    readouts = {}
    for section, label in (("posthoc_mask_readout", "prototype_majority"),
                           ("linear_mask_readout", "prototype_distance_linear"),
                           ("direct_intensity_readout_control", "direct_intensity_linear")):
        cm = sum_confusion(section, "heldout_patch_confusion_matrix_rows_true_cols_predicted")
        readouts[label] = {"confusion_matrix_rows_true_cols_predicted": cm.tolist(),
                           "dice": dice_from_cm(cm)}

    group_masked_prediction = {}
    for modality in modalities:
        modality_rows = [r["whole_modality_group_masked_prediction"][modality] for r in runs]
        n = sum(row["n_patches"] for row in modality_rows)
        feature_names = modality_rows[0]["features"]
        group_masked_prediction[modality] = {"n_patches": n, "features": {}}
        for feature in feature_names:
            model_mse = sum(row["features"][feature]["mycelia_mse"]*row["n_patches"]
                            for row in modality_rows)/max(1,n)
            ridge_mse = sum(row["features"][feature]["ridge_mse"]*row["n_patches"]
                            for row in modality_rows)/max(1,n)
            group_masked_prediction[modality]["features"][feature] = {
                "mycelia_mse": model_mse, "mycelia_rmse": float(np.sqrt(model_mse)),
                "ridge_mse": ridge_mse, "ridge_rmse": float(np.sqrt(ridge_mse)),
            }
        group_masked_prediction[modality]["selected_local_specialist_weight_by_fold"] = [
            run["whole_modality_group_masked_prediction"][modality]["selected_local_specialist_weight"]
            for run in runs
        ]
        if runs[0].get("visual_representation") == "dct8":
            ordered = [group_masked_prediction[modality]["features"][f"dct_{z}{y}{x}"]
                       for z, y, x in _DCT8_FREQUENCIES]
            mycelia_projection_mse = dct8_projection_mse([row["mycelia_mse"] for row in ordered])
            ridge_projection_mse = dct8_projection_mse([row["ridge_mse"] for row in ordered])
            group_masked_prediction[modality]["low_frequency_patch_projection"] = {
                "mycelia_mse_per_voxel": mycelia_projection_mse,
                "mycelia_rmse_per_voxel": float(np.sqrt(mycelia_projection_mse)),
                "ridge_mse_per_voxel": ridge_projection_mse,
                "ridge_rmse_per_voxel": float(np.sqrt(ridge_projection_mse)),
                "interpretation": "inverse orthonormal 3D patch-DCT error over the eight modeled low-frequency modes only; omitted high-frequency detail is not predicted",
            }

    output = {
        "protocol": "5-fold cross-validation grouped by patient; every patient held out once with all longitudinal visits together",
        "visual_representation": runs[0].get("visual_representation", "summary"),
        "seed": runs[0]["seed"], "folds": args.folds,
        "total_unique_patients": len(set(test_patients)),
        "heldout_visits": total_test_visits,
        "heldout_patch_counts_by_fold": patch_counts,
        "patient_overlap_across_heldout_folds": False,
        "encoded_cross_split_duplicate_count_each_fold": 0,
        "heldout_prediction": prediction,
        "native_volumetric_mycelia_prediction": native_prediction,
        "native_local_predictive_prediction": native_local,
        "native_predictive_route_calibration_by_fold": [
            r["native_volumetric_mycelia"]["predictive_route_calibration"] for r in runs],
        "native_route_local_blend_prediction": native_blend,
        "native_blend_calibration_by_fold": [r["native_volumetric_mycelia"]["train_only_calibration"] for r in runs],
        "native_memory_by_fold": [r["native_volumetric_mycelia"]["training_state"] for r in runs],
        "mask_readouts": readouts,
        "whole_modality_group_masked_prediction": group_masked_prediction,
        "grouped_predictor_context_scales": runs[0].get("grouped_predictor_context_scales", [1]),
        "grouped_specialist_influence_calibration_by_fold": [
            run.get("grouped_specialist_influence_calibration") for run in runs],
        "unsupervised_cluster_alignment": {
            "mean_ari": float(np.mean([r["heldout_cluster_alignment_posthoc_only"]["adjusted_rand_index"] for r in runs])),
            "fold_ari": [r["heldout_cluster_alignment_posthoc_only"]["adjusted_rand_index"] for r in runs],
            "mean_nmi": float(np.mean([r["heldout_cluster_alignment_posthoc_only"]["normalized_mutual_information"] for r in runs])),
            "fold_nmi": [r["heldout_cluster_alignment_posthoc_only"]["normalized_mutual_information"] for r in runs],
        },
        "multiscale_visual_cluster_alignment": {
            "mean_ari": float(np.mean([r["heldout_multiscale_visual_cluster_alignment_posthoc_only"]["adjusted_rand_index"] for r in runs])),
            "fold_ari": [r["heldout_multiscale_visual_cluster_alignment_posthoc_only"]["adjusted_rand_index"] for r in runs],
            "mean_nmi": float(np.mean([r["heldout_multiscale_visual_cluster_alignment_posthoc_only"]["normalized_mutual_information"] for r in runs])),
            "fold_nmi": [r["heldout_multiscale_visual_cluster_alignment_posthoc_only"]["normalized_mutual_information"] for r in runs],
            "features": runs[0]["visual_descriptor_features"],
        },
        "mycelia_visual_colony_alignment": {
            "mean_ari": float(np.mean([r["heldout_mycelia_visual_colony_alignment_posthoc_only"]["adjusted_rand_index"] for r in runs])),
            "fold_ari": [r["heldout_mycelia_visual_colony_alignment_posthoc_only"]["adjusted_rand_index"] for r in runs],
            "mean_nmi": float(np.mean([r["heldout_mycelia_visual_colony_alignment_posthoc_only"]["normalized_mutual_information"] for r in runs])),
            "fold_nmi": [r["heldout_mycelia_visual_colony_alignment_posthoc_only"]["normalized_mutual_information"] for r in runs],
            "memory_state_by_fold": [r["mycelia_visual_colony_state"] for r in runs],
        },
        "patient_visit_dice": {
            "prototype_majority": [x for r in runs for x in r["posthoc_mask_readout"]["heldout_patient_visit_whole_tumor_patch_dice"]],
            "prototype_distance_linear": [x for r in runs for x in r["linear_mask_readout"]["heldout_patient_visit_whole_tumor_patch_dice"]],
        },
        "limitations": runs[0]["limitations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"total_unique_patients": output["total_unique_patients"],
                      "heldout_visits": total_test_visits,
                      "heldout_prediction": prediction,
                      "native_volumetric_mycelia_prediction": native_prediction,
                      "native_local_predictive_prediction": native_local,
                      "native_route_local_blend_prediction": native_blend,
                      "mask_readouts": readouts,
                      "whole_modality_group_masked_prediction": group_masked_prediction,
                      "unsupervised_cluster_alignment": output["unsupervised_cluster_alignment"],
                      "multiscale_visual_cluster_alignment": output["multiscale_visual_cluster_alignment"],
                      "mycelia_visual_colony_alignment": output["mycelia_visual_colony_alignment"]}, indent=2))


if __name__ == "__main__":
    main()
