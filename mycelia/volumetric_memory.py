"""Label-free competitive colony memory for registered multimodal volumes.

This is an explicit bio-inspired computational model: sensory patch vectors
compete for a finite set of local specialists, while six-direction colony
transitions accumulate route strength. The mechanisms are hypotheses and do
not claim that fungi perform MRI-like vector quantization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import json

import numpy as np


# (source offset z,y,x, direction from source toward target)
_NEIGHBORS = (
    (-1, 0, 0, 0),  # +z
    (1, 0, 0, 1),   # -z
    (0, -1, 0, 2),  # +y
    (0, 1, 0, 3),   # -y
    (0, 0, -1, 4),  # +x
    (0, 0, 1, 5),   # -x
)


@dataclass(frozen=True)
class VolumetricMemoryConfig:
    max_colonies: int = 64
    novelty_threshold: float = 0.08
    learning_rate: float = 0.04
    context_bandwidth: float = 0.18
    activity_threshold: float = 0.01
    predictive_ridge: float = 10.0

    def __post_init__(self):
        if isinstance(self.max_colonies, bool) or not isinstance(self.max_colonies, int) or self.max_colonies < 2:
            raise ValueError('max_colonies must be an integer >= 2')
        for key in ('novelty_threshold', 'learning_rate', 'context_bandwidth', 'activity_threshold',
                    'predictive_ridge'):
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be finite and positive')
        if self.learning_rate > 1 or self.activity_threshold >= 1:
            raise ValueError('learning_rate must be <= 1 and activity_threshold < 1')


class VolumetricMycelium:
    """Finite prototype colonies plus locally reinforced 3D directional routes.

    Fitting accepts only image volumes. Expert labels, masks, patients, and
    diagnoses are not inputs to the learner.
    """

    def __init__(self, config=None):
        self.config = config or VolumetricMemoryConfig()
        self.centers = np.empty((0, 0), dtype=float)
        self.support = np.empty(0, dtype=np.int64)
        self.routes = {}
        self.route_rows = {}
        self.route_updates = 0
        self.images_seen = 0
        self.patches_seen = 0
        self.training_mean = None
        self.predictive_support = np.empty((0, 0), dtype=np.int64)
        self.predictive_coefficients = np.empty((0, 0, 0), dtype=float)
        self.global_predictive_coefficients = np.empty((0, 0), dtype=float)
        self._training_sum = None
        self._training_count = 0
        self.predictive_xtx = np.empty((0, 0, 0, 0), dtype=float)
        self.predictive_xty = np.empty((0, 0, 0), dtype=float)
        self.global_predictive_xtx = np.empty((0, 0, 0), dtype=float)
        self.global_predictive_xty = np.empty((0, 0), dtype=float)
        self.predictive_routes = []
        self.predictive_route_rows = []
        self.predictive_route_updates = np.zeros(0, dtype=np.int64)
        self.predictive_maps_enabled = True

    def _validate_volume(self, volume):
        value = np.asarray(volume, dtype=float)
        if (value.ndim != 4 or min(value.shape[:3]) < 1 or value.shape[-1] < 2 or
                not np.all(np.isfinite(value))):
            raise ValueError('volume must be finite ZxYxXxC data with C >= 2')
        if self.centers.shape[1:] and value.shape[-1] != self.centers.shape[1]:
            raise ValueError('volume modality count differs from fitted colony vectors')
        return value

    def fit(self, volumes, *, seed=0, fit_predictors=True):
        """Online competitive learning and post-fit six-neighbor route reinforcement."""
        if not isinstance(fit_predictors, (bool, np.bool_)):
            raise ValueError('fit_predictors must be boolean')
        volumes = [self._validate_volume(v) if self.centers.size else np.asarray(v, dtype=float)
                   for v in volumes]
        if not volumes:
            raise ValueError('at least one training volume is required')
        checked = [self._validate_training_volume(v) for v in volumes]
        channels = checked[0].shape[-1]
        if any(v.shape[-1] != channels for v in checked):
            raise ValueError('training volumes must have the same modality count')
        active_vectors = np.concatenate([
            volume[np.max(np.abs(volume), axis=-1) > self.config.activity_threshold]
            for volume in checked
        ], axis=0)
        if not len(active_vectors):
            raise ValueError('training volumes contain no active image patches')
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(active_vectors))
        self.centers = np.empty((0, channels), dtype=float)
        self.support = np.empty(0, dtype=np.int64)
        self.patches_seen = 0
        for index in order:
            sample = active_vectors[index]
            if not len(self.centers):
                self.centers = sample[None, :].copy()
                self.support = np.ones(1, dtype=np.int64)
            else:
                distances = np.sqrt(np.mean((self.centers-sample[None, :])**2, axis=1))
                winner = int(np.argmin(distances))
                if (distances[winner] > self.config.novelty_threshold and
                        len(self.centers) < self.config.max_colonies):
                    self.centers = np.vstack((self.centers, sample))
                    self.support = np.append(self.support, 1)
                else:
                    self.support[winner] += 1
                    rate = self.config.learning_rate / math.sqrt(float(self.support[winner]))
                    self.centers[winner] += rate * (sample-self.centers[winner])
            self.patches_seen += 1
        self.images_seen = len(checked)
        self.training_mean = active_vectors.mean(axis=0)
        self._training_sum = active_vectors.sum(axis=0)
        self._training_count = len(active_vectors)
        self.routes = {}
        self.route_rows = {}
        self.route_updates = 0
        for volume in checked:
            assignments = self.assignments(volume)
            self._reinforce_routes(assignments)
        rows = {}
        for (source, direction, target), weight in self.routes.items():
            rows.setdefault((source, direction), {})[target] = weight
        self.route_rows = rows
        self.predictive_maps_enabled = bool(fit_predictors)
        if self.predictive_maps_enabled:
            self._fit_local_predictors(checked)
        else:
            self._clear_predictive_maps(channels, len(self.centers))
        return self

    def _clear_predictive_maps(self, channels, colonies):
        """Keep topology-only fits lightweight when another learner owns prediction."""
        self.predictive_support = np.zeros((channels, colonies), dtype=np.int64)
        self.predictive_coefficients = np.empty((0, 0, 0), dtype=float)
        self.global_predictive_coefficients = np.empty((0, 0), dtype=float)
        self.predictive_xtx = np.empty((0, 0, 0, 0), dtype=float)
        self.predictive_xty = np.empty((0, 0, 0), dtype=float)
        self.global_predictive_xtx = np.empty((0, 0, 0), dtype=float)
        self.global_predictive_xty = np.empty((0, 0), dtype=float)
        self.predictive_routes = [dict() for _ in range(channels)]
        self.predictive_route_rows = [dict() for _ in range(channels)]
        self.predictive_route_updates = np.zeros(channels, dtype=np.int64)

    def partial_fit(self, volume, *, seed=0):
        """Absorb one new image volume while retaining previously learned state.

        New visual motifs recruit unused colony slots. Once the colony budget
        is full, active colonies continue to adapt by a support-scaled local
        update. Route counts and predictive co-activation statistics persist
        across calls, making this a genuine streaming interface rather than a
        batch refit disguised as adaptation.
        """
        value = self._validate_training_volume(volume)
        if not self.centers.size:
            return self.fit([value], seed=seed)
        if value.shape[-1] != self.centers.shape[1]:
            raise ValueError('volume modality count differs from fitted colony vectors')
        active_vectors = value[np.max(np.abs(value), axis=-1) > self.config.activity_threshold]
        if not len(active_vectors):
            raise ValueError('volume contains no active image patches')
        for sample in active_vectors:
            distances = np.sqrt(np.mean((self.centers-sample[None, :])**2, axis=1))
            winner = int(np.argmin(distances))
            if (distances[winner] > self.config.novelty_threshold and
                    len(self.centers) < self.config.max_colonies):
                self.centers = np.vstack((self.centers, sample))
                self.support = np.append(self.support, 1)
                if self.predictive_maps_enabled:
                    self._expand_predictive_capacity()
            else:
                self.support[winner] += 1
                rate = self.config.learning_rate / math.sqrt(float(self.support[winner]))
                self.centers[winner] += rate * (sample-self.centers[winner])
            self.patches_seen += 1
        self._training_sum += active_vectors.sum(axis=0)
        self._training_count += len(active_vectors)
        self.training_mean = self._training_sum / self._training_count
        self.images_seen += 1
        self._reinforce_routes(self.assignments(value))
        self._rebuild_route_rows()
        if self.predictive_maps_enabled:
            self._accumulate_predictive_volume(value)
            self._refresh_predictive_coefficients()
        self.validate()
        return self

    def _rebuild_route_rows(self):
        rows = {}
        for (source, direction, target), weight in self.routes.items():
            rows.setdefault((source, direction), {})[target] = weight
        self.route_rows = rows

    def _expand_predictive_capacity(self):
        """Append zeroed local Hebbian statistics when a new colony branches."""
        channels = self.centers.shape[1]
        feature_count = 7 * channels
        if self.predictive_xtx.size == 0:
            self.predictive_xtx = np.zeros((channels, 0, feature_count, feature_count))
            self.predictive_xty = np.zeros((channels, 0, feature_count))
            self.predictive_support = np.zeros((channels, 0), dtype=np.int64)
        self.predictive_xtx = np.concatenate(
            (self.predictive_xtx, np.zeros((channels, 1, feature_count, feature_count))), axis=1)
        self.predictive_xty = np.concatenate(
            (self.predictive_xty, np.zeros((channels, 1, feature_count))), axis=1)
        self.predictive_support = np.pad(self.predictive_support, ((0, 0), (0, 1)))
        if self.predictive_coefficients.size:
            self.predictive_coefficients = np.concatenate(
                (self.predictive_coefficients,
                 self.global_predictive_coefficients[:, None, :]), axis=1)

    @staticmethod
    def _context_matrix(volume, modality):
        """Six-neighbor sensory context plus the other center modalities."""
        channels = volume.shape[-1]
        neighbors = []
        for axis in range(3):
            for delta in (-1, 1):
                shifted = np.zeros_like(volume)
                dst = [slice(None)] * 4
                src = [slice(None)] * 4
                if delta == 1:
                    dst[axis], src[axis] = slice(0, -1), slice(1, None)
                else:
                    dst[axis], src[axis] = slice(1, None), slice(0, -1)
                shifted[tuple(dst)] = volume[tuple(src)]
                neighbors.append(shifted.reshape(-1, channels))
        center = volume.reshape(-1, channels)
        cues = np.column_stack((*neighbors, np.delete(center, modality, axis=1)))
        return np.column_stack((cues, np.ones(len(cues), dtype=float)))

    def _fit_local_predictors(self, volumes):
        """Learn local Hebbian predictive maps from unlabeled image context.

        Each colony accumulates context covariance and context/target
        co-activation for each withheld modality. Solving these sufficient
        statistics gives a small regularized local linear predictor; no image
        labels, masks, or backpropagation are involved.
        """
        channels = self.centers.shape[1]
        colonies = len(self.centers)
        feature_count = 6 * channels + channels - 1 + 1  # six neighbors, visible center, bias
        self.global_predictive_xtx = np.zeros((channels, feature_count, feature_count), dtype=float)
        self.global_predictive_xty = np.zeros((channels, feature_count), dtype=float)
        self.predictive_xtx = np.zeros((channels, colonies, feature_count, feature_count), dtype=float)
        self.predictive_xty = np.zeros((channels, colonies, feature_count), dtype=float)
        self.predictive_support = np.zeros((channels, colonies), dtype=np.int64)
        self.predictive_routes = [dict() for _ in range(channels)]
        self.predictive_route_rows = [dict() for _ in range(channels)]
        self.predictive_route_updates = np.zeros(channels, dtype=np.int64)
        for volume in volumes:
            self._accumulate_predictive_volume(volume)
        self._refresh_predictive_coefficients()

    def _accumulate_predictive_volume(self, volume):
        channels = self.centers.shape[1]
        active = np.max(np.abs(volume), axis=-1).reshape(-1) > self.config.activity_threshold
        targets = volume.reshape(-1, channels)
        for modality in range(channels):
            cue_volume = volume.copy()
            cue_volume[..., modality] = self.training_mean[modality]
            colony_grid = self.assignments(cue_volume).reshape(-1)
            self._reinforce_predictive_routes(colony_grid.reshape(volume.shape[:3]), modality)
            valid = active & (colony_grid >= 0)
            if not valid.any():
                continue
            context = self._context_matrix(volume, modality)[valid]
            target = targets[valid, modality]
            colony_ids = colony_grid[valid]
            self.global_predictive_xtx[modality] += context.T @ context
            self.global_predictive_xty[modality] += context.T @ target
            for colony in np.unique(colony_ids):
                selected = colony_ids == colony
                local_context = context[selected]
                local_target = target[selected]
                self.predictive_xtx[modality, colony] += local_context.T @ local_context
                self.predictive_xty[modality, colony] += local_context.T @ local_target
                self.predictive_support[modality, colony] += int(selected.sum())
        self._rebuild_predictive_route_rows()

    def _reinforce_predictive_routes(self, assignments, modality):
        shape = assignments.shape
        for coordinate in np.ndindex(shape):
            target = int(assignments[coordinate])
            if target < 0:
                continue
            z, y, x = coordinate
            for dz, dy, dx, direction in _NEIGHBORS:
                source_coordinate = (z+dz, y+dy, x+dx)
                if any(not 0 <= source_coordinate[d] < shape[d] for d in range(3)):
                    continue
                source = int(assignments[source_coordinate])
                if source < 0:
                    continue
                key = (source, direction, target)
                routes = self.predictive_routes[modality]
                routes[key] = routes.get(key, 0.0) + 1.0
                self.predictive_route_updates[modality] += 1

    def _rebuild_predictive_route_rows(self):
        rows = []
        for routes in self.predictive_routes:
            modality_rows = {}
            for (source, direction, target), weight in routes.items():
                modality_rows.setdefault((source, direction), {})[target] = weight
            rows.append(modality_rows)
        self.predictive_route_rows = rows

    def _refresh_predictive_coefficients(self):
        channels = self.centers.shape[1]
        colonies = len(self.centers)
        feature_count = 7 * channels
        penalty = np.eye(feature_count) * self.config.predictive_ridge
        penalty[-1, -1] = 0.0  # leave the bias unpenalized
        self.global_predictive_coefficients = np.stack([
            np.linalg.solve(self.global_predictive_xtx[m] + penalty,
                            self.global_predictive_xty[m])
            for m in range(channels)
        ])
        self.predictive_coefficients = np.empty((channels, colonies, feature_count), dtype=float)
        for modality in range(channels):
            for colony in range(colonies):
                count = int(self.predictive_support[modality, colony])
                if count == 0:
                    self.predictive_coefficients[modality, colony] = self.global_predictive_coefficients[modality]
                    continue
                local = np.linalg.solve(self.predictive_xtx[modality, colony] + penalty,
                                        self.predictive_xty[modality, colony])
                # Sparse colonies defer to shared experience; established
                # specialists gradually gain influence as evidence accrues.
                confidence = count / (count + 64.0)
                self.predictive_coefficients[modality, colony] = (
                    confidence * local + (1.0-confidence) * self.global_predictive_coefficients[modality]
                )

    @staticmethod
    def _validate_training_volume(volume):
        value = np.asarray(volume, dtype=float)
        if (value.ndim != 4 or min(value.shape[:3]) < 1 or value.shape[-1] < 2 or
                not np.all(np.isfinite(value))):
            raise ValueError('training volume must be finite ZxYxXxC data with C >= 2')
        return value

    @property
    def colony_count(self):
        return len(self.centers)

    @property
    def route_count(self):
        return len(self.routes)

    def assignments(self, volume, *, masked_coordinate=None, masked_channel=None):
        """Map patches to colonies; optional center channel can be hidden first."""
        volume = self._validate_volume(volume)
        cues = volume.reshape(-1, volume.shape[-1]).copy()
        if masked_coordinate is not None:
            if masked_channel is None or not 0 <= masked_channel < volume.shape[-1]:
                raise ValueError('masked_channel is required and must be a valid modality index')
            coordinate = tuple(masked_coordinate)
            if len(coordinate) != 3 or any(not 0 <= coordinate[d] < volume.shape[d] for d in range(3)):
                raise ValueError('masked_coordinate is outside the volume')
            flat_index = np.ravel_multi_index(coordinate, volume.shape[:3])
            cues[flat_index, masked_channel] = self.training_mean[masked_channel]
        active = np.max(cues, axis=1) > self.config.activity_threshold
        result = np.full(len(cues), -1, dtype=int)
        if active.any() and len(self.centers):
            distance = np.mean((cues[active, None, :]-self.centers[None, :, :])**2, axis=-1)
            result[active] = np.argmin(distance, axis=1)
        return result.reshape(volume.shape[:3])

    def _reinforce_routes(self, assignments):
        shape = assignments.shape
        for coordinate in np.ndindex(shape):
            target = int(assignments[coordinate])
            if target < 0:
                continue
            z, y, x = coordinate
            for dz, dy, dx, direction in _NEIGHBORS:
                source_coordinate = (z+dz, y+dy, x+dx)
                if any(not 0 <= source_coordinate[d] < shape[d] for d in range(3)):
                    continue
                source = int(assignments[source_coordinate])
                if source < 0:
                    continue
                key = (source, direction, target)
                # Repeated local co-activation reinforces the directional route.
                self.routes[key] = self.routes.get(key, 0.0)+1.0
                self.route_updates += 1

    def predict_masked(self, volume, coordinate, modality, *, assignments=None):
        """Predict one hidden modality from other modalities and 3D neighbor routes."""
        volume = self._validate_volume(volume)
        if isinstance(modality, bool) or not isinstance(modality, int) or not 0 <= modality < volume.shape[-1]:
            raise ValueError('modality index outside the input channel range')
        coordinate = tuple(coordinate)
        if len(coordinate) != 3 or any(not 0 <= coordinate[d] < volume.shape[d] for d in range(3)):
            raise ValueError('coordinate outside volume')
        z, y, x = coordinate
        if assignments is None:
            assignments = self.assignments(volume, masked_coordinate=coordinate,
                                           masked_channel=modality)
        else:
            assignments = np.asarray(assignments, dtype=int)
            if assignments.shape != volume.shape[:3]:
                raise ValueError('assignment grid must match volume spatial shape')
            # Only adjacent positions are read below. The target's own colony
            # ID is never consulted, even when this cached map contains one.
        votes = {}
        used_sources = 0
        for dz, dy, dx, direction in _NEIGHBORS:
            source_coordinate = (z+dz, y+dy, x+dx)
            if any(not 0 <= source_coordinate[d] < volume.shape[d] for d in range(3)):
                continue
            source = int(assignments[source_coordinate])
            if source < 0:
                continue
            row = self.route_rows.get((source, direction), {})
            total = sum(row.values())
            if total <= 0:
                continue
            used_sources += 1
            for target, weight in row.items():
                votes[target] = votes.get(target, 0.0)+weight/total
        if not votes:
            return {'prediction': float(self.training_mean[modality]),
                    'context_sources': 0, 'candidate_count': 0}
        candidates = np.asarray(list(votes), dtype=int)
        priors = np.asarray([votes[int(i)] for i in candidates], dtype=float)
        visible = np.arange(volume.shape[-1]) != modality
        residual = self.centers[candidates][:, visible]-volume[coordinate][visible][None, :]
        distance = np.mean(residual*residual, axis=1)
        affinity = np.exp(-distance/(2*self.config.context_bandwidth**2))
        weights = priors*affinity
        if not np.isfinite(weights.sum()) or weights.sum() <= 1e-15:
            weights = priors
        weights /= weights.sum()
        estimate = float(np.dot(weights, self.centers[candidates, modality]))
        return {'prediction': estimate, 'context_sources': used_sources,
                'candidate_count': len(candidates)}

    def predict_volume(self, volume):
        """Return full patch-grid predictions, using channel-wise masked prediction."""
        if not self.predictive_maps_enabled:
            raise RuntimeError('this topology-only memory was fitted without predictive maps')
        volume = self._validate_volume(volume)
        assignments = self.assignments(volume)
        predictions = np.full(volume.shape, np.nan, dtype=float)
        source_counts = np.zeros((*volume.shape[:3], volume.shape[-1]), dtype=np.uint8)
        active = np.max(np.abs(volume), axis=-1) > self.config.activity_threshold
        for coordinate in zip(*np.where(active)):
            for modality in range(volume.shape[-1]):
                result = self.predict_masked(volume, coordinate, modality,
                                             assignments=assignments)
                predictions[coordinate+(modality,)] = result['prediction']
                source_counts[coordinate+(modality,)] = result['context_sources']
        return {'predictions': predictions, 'context_sources': source_counts,
                'active_patches': int(active.sum()),
                'route_coverage': float(np.mean(source_counts[active] > 0)) if active.any() else 0.0}

    def predict_local_volume(self, volume, *, route_weight=0.0):
        """Predict active patches from local context and optional route voting.

        Route candidates come only from neighboring colony assignments. The
        target center is assigned after withholding its requested modality;
        its modality value is never used to select a local specialist.
        """
        if not self.predictive_maps_enabled:
            raise RuntimeError('this topology-only memory was fitted without predictive maps')
        volume = self._validate_volume(volume)
        channels = volume.shape[-1]
        if isinstance(route_weight, bool):
            raise ValueError('route_weight must not be boolean')
        raw_route_weights = np.asarray(route_weight)
        if raw_route_weights.dtype.kind == 'b':
            raise ValueError('route_weight must not be boolean')
        route_weights = np.asarray(route_weight, dtype=float)
        if route_weights.ndim == 0:
            route_weights = np.full(channels, float(route_weights))
        if (route_weights.shape != (channels,) or not np.all(np.isfinite(route_weights)) or
                np.any((route_weights < 0) | (route_weights > 1))):
            raise ValueError('route_weight must be a scalar or one finite [0,1] weight per modality')
        flat = volume.reshape(-1, channels)
        active = np.max(np.abs(volume), axis=-1).reshape(-1) > self.config.activity_threshold
        predictions = np.full((len(flat), channels), np.nan, dtype=float)
        route_predictions = np.full((len(flat), channels), np.nan, dtype=float)
        colony_ids = np.full((len(flat), channels), -1, dtype=int)
        route_sources = np.zeros((len(flat), channels), dtype=np.uint8)
        for modality in range(channels):
            cue_volume = volume.copy()
            cue_volume[..., modality] = self.training_mean[modality]
            assignments_grid = self.assignments(cue_volume)
            source_assignments = assignments_grid
            assignments = assignments_grid.reshape(-1)
            valid = active & (assignments >= 0)
            if not valid.any():
                continue
            features = self._context_matrix(volume, modality)[valid]
            ids = assignments[valid]
            predictions[valid, modality] = np.einsum(
                'ij,ij->i', features, self.predictive_coefficients[modality, ids]
            )
            colony_ids[valid, modality] = ids
            modality_route_weight = route_weights[modality]
            if modality_route_weight > 0:
                coords = np.flatnonzero(valid)
                spatial = np.column_stack(np.unravel_index(coords, volume.shape[:3]))
                for row_index, (flat_index, coordinate) in enumerate(zip(coords, spatial)):
                    votes = {}
                    z, y, x = (int(c) for c in coordinate)
                    for dz, dy, dx, direction in _NEIGHBORS:
                        source_coordinate = (z+dz, y+dy, x+dx)
                        if any(not 0 <= source_coordinate[d] < volume.shape[d] for d in range(3)):
                            continue
                        source = int(source_assignments[source_coordinate])
                        if source < 0:
                            continue
                        route = self.predictive_route_rows[modality].get((source, direction), {})
                        total = sum(route.values())
                        if total <= 0:
                            continue
                        route_sources[flat_index, modality] += 1
                        for target, strength in route.items():
                            votes[target] = votes.get(target, 0.0) + strength / total
                    if votes:
                        candidate_ids = np.asarray(list(votes), dtype=int)
                        candidate_weights = np.asarray([votes[i] for i in candidate_ids], dtype=float)
                        candidate_weights /= candidate_weights.sum()
                        route_predictions[flat_index, modality] = np.dot(
                            candidate_weights,
                            self.predictive_coefficients[modality, candidate_ids] @ features[row_index],
                        )
                    else:
                        route_predictions[flat_index, modality] = predictions[flat_index, modality]
        combined = predictions.copy()
        for modality, weight in enumerate(route_weights):
            if weight > 0:
                combined[..., modality] = ((1.0-weight)*predictions[..., modality] +
                                           weight*route_predictions[..., modality])
        return {'predictions': combined.reshape(volume.shape),
                'local_predictions': predictions.reshape(volume.shape),
                'route_predictions': route_predictions.reshape(volume.shape),
                'context_colonies': colony_ids.reshape((*volume.shape[:3], channels)),
                'route_context_sources': route_sources.reshape((*volume.shape[:3], channels)),
                'active_patches': int(active.sum())}

    def validate(self):
        if self.centers.ndim != 2 or not len(self.centers) or self.support.shape != (len(self.centers),):
            raise ValueError('invalid or empty colony bank')
        if len(self.centers) > self.config.max_colonies or np.any(self.support < 1):
            raise ValueError('colony capacity or support invariant failed')
        if any(source < 0 or source >= len(self.centers) or target < 0 or target >= len(self.centers) or
               not 0 <= direction < 6 or not math.isfinite(weight) or weight <= 0
               for (source, direction, target), weight in self.routes.items()):
            raise ValueError('invalid route state')
        if sum(self.support) != self.patches_seen or sum(self.routes.values()) != self.route_updates:
            raise ValueError('colony support or route reinforcement totals are inconsistent')
        if not self.predictive_maps_enabled:
            if any(array.size for array in (
                    self.predictive_coefficients, self.global_predictive_coefficients,
                    self.predictive_xtx, self.predictive_xty,
                    self.global_predictive_xtx, self.global_predictive_xty)):
                raise ValueError('topology-only memory unexpectedly contains predictive maps')
            return {'colonies': len(self.centers), 'patches_seen': self.patches_seen,
                    'routes': len(self.routes), 'route_updates': self.route_updates,
                    'images_seen': self.images_seen,
                    'predictive_maps_enabled': False}
        expected_features = 7 * self.centers.shape[1]
        if self.predictive_coefficients.shape != (self.centers.shape[1], len(self.centers), expected_features):
            raise ValueError('local predictive coefficient dimensions are inconsistent')
        if self.global_predictive_coefficients.shape != (self.centers.shape[1], expected_features):
            raise ValueError('global predictive coefficient dimensions are inconsistent')
        if self.predictive_support.shape != (self.centers.shape[1], len(self.centers)):
            raise ValueError('local predictive support dimensions are inconsistent')
        if self.predictive_xtx.shape != (self.centers.shape[1], len(self.centers), expected_features, expected_features):
            raise ValueError('local predictive covariance dimensions are inconsistent')
        if self.predictive_xty.shape != (self.centers.shape[1], len(self.centers), expected_features):
            raise ValueError('local predictive association dimensions are inconsistent')
        if self.global_predictive_xtx.shape != (self.centers.shape[1], expected_features, expected_features):
            raise ValueError('global predictive covariance dimensions are inconsistent')
        if self.global_predictive_xty.shape != (self.centers.shape[1], expected_features):
            raise ValueError('global predictive association dimensions are inconsistent')
        if (len(self.predictive_routes) != self.centers.shape[1] or
                self.predictive_route_updates.shape != (self.centers.shape[1],)):
            raise ValueError('modality-specific route state dimensions are inconsistent')
        for modality, routes in enumerate(self.predictive_routes):
            if any(source < 0 or source >= len(self.centers) or target < 0 or target >= len(self.centers) or
                   not 0 <= direction < 6 or not math.isfinite(weight) or weight <= 0
                   for (source, direction, target), weight in routes.items()):
                raise ValueError(f'invalid modality-specific route state for modality {modality}')
            if sum(routes.values()) != self.predictive_route_updates[modality]:
                raise ValueError('modality-specific route reinforcement totals are inconsistent')
        if (self._training_sum is None or self._training_sum.shape != self.training_mean.shape or
                self._training_count != self.patches_seen or
                not np.allclose(self._training_sum / self._training_count, self.training_mean)):
            raise ValueError('running sensory mean is inconsistent with seen patches')
        return {'colonies': len(self.centers), 'patches_seen': self.patches_seen,
                'routes': len(self.routes), 'route_updates': self.route_updates,
                'images_seen': self.images_seen}

    def save(self, path):
        """Persist learned colonies and routes without pickled Python objects."""
        self.validate()
        keys = np.asarray(sorted(self.routes), dtype=np.int64).reshape(-1, 3)
        weights = np.asarray([self.routes[tuple(key)] for key in keys], dtype=float)
        predictive_items = [(modality, *key, weight)
                            for modality, routes in enumerate(self.predictive_routes)
                            for key, weight in sorted(routes.items())]
        predictive_keys = np.asarray([item[:4] for item in predictive_items], dtype=np.int64).reshape(-1, 4)
        predictive_weights = np.asarray([item[4] for item in predictive_items], dtype=float)
        np.savez_compressed(
            Path(path), centers=self.centers, support=self.support,
            route_keys=keys, route_weights=weights,
            training_mean=self.training_mean,
            config=np.asarray(json.dumps(self.config.__dict__, sort_keys=True)),
            images_seen=np.asarray(self.images_seen), patches_seen=np.asarray(self.patches_seen),
            route_updates=np.asarray(self.route_updates),
            predictive_maps_enabled=np.asarray(self.predictive_maps_enabled),
            predictive_support=self.predictive_support,
            predictive_coefficients=self.predictive_coefficients,
            global_predictive_coefficients=self.global_predictive_coefficients,
            predictive_xtx=self.predictive_xtx,
            predictive_xty=self.predictive_xty,
            global_predictive_xtx=self.global_predictive_xtx,
            global_predictive_xty=self.global_predictive_xty,
            training_sum=self._training_sum,
            training_count=np.asarray(self._training_count),
            predictive_route_keys=predictive_keys,
            predictive_route_weights=predictive_weights,
            predictive_route_updates=self.predictive_route_updates,
        )

    @classmethod
    def load(cls, path):
        """Restore a validated state created by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as state:
            config = VolumetricMemoryConfig(**json.loads(str(state['config'].item())))
            model = cls(config)
            model.centers = state['centers'].astype(float)
            model.support = state['support'].astype(np.int64)
            model.training_mean = state['training_mean'].astype(float)
            keys = state['route_keys'].astype(np.int64)
            weights = state['route_weights'].astype(float)
            if keys.shape != (len(weights), 3):
                raise ValueError('route key and weight arrays do not align')
            model.routes = {tuple(int(v) for v in key): float(weight)
                            for key, weight in zip(keys, weights)}
            rows = {}
            for (source, direction, target), weight in model.routes.items():
                rows.setdefault((source, direction), {})[target] = weight
            model.route_rows = rows
            model.images_seen = int(state['images_seen'])
            model.patches_seen = int(state['patches_seen'])
            model.route_updates = int(state['route_updates'])
            model.predictive_maps_enabled = bool(state['predictive_maps_enabled']) if 'predictive_maps_enabled' in state else True
            model.predictive_xtx = state['predictive_xtx'].astype(float)
            model.predictive_xty = state['predictive_xty'].astype(float)
            model.global_predictive_xtx = state['global_predictive_xtx'].astype(float)
            model.global_predictive_xty = state['global_predictive_xty'].astype(float)
            model._training_sum = state['training_sum'].astype(float)
            model._training_count = int(state['training_count'])
            predictive_keys = state['predictive_route_keys'].astype(np.int64)
            predictive_weights = state['predictive_route_weights'].astype(float)
            channels = model.centers.shape[1]
            if predictive_keys.shape != (len(predictive_weights), 4):
                raise ValueError('predictive route keys and weights do not align')
            model.predictive_routes = [dict() for _ in range(channels)]
            for key, weight in zip(predictive_keys, predictive_weights):
                modality, source, direction, target = (int(v) for v in key)
                model.predictive_routes[modality][(source, direction, target)] = float(weight)
            model.predictive_route_updates = state['predictive_route_updates'].astype(np.int64)
            model._rebuild_predictive_route_rows()
            if 'predictive_support' in state:
                model.predictive_support = state['predictive_support'].astype(np.int64)
                model.predictive_coefficients = state['predictive_coefficients'].astype(float)
                model.global_predictive_coefficients = state['global_predictive_coefficients'].astype(float)
        model.validate()
        return model


class GroupedPredictiveMemory:
    """Colony-local prediction when a complete feature group is hidden.

    A group can encode all cues from one acquisition sequence (for example,
    patch mean, texture and gradient). During both fitting and inference every
    center cue in that group is withheld together; neighboring voxels remain
    observable at multiple spatial radii. The local maps learn context/target
    co-activation from images only and condition specialists on the supplied
    competitive colony memory. Specialist influence can be gated toward shared
    experience when patient-held-out calibration does not support a local map.
    """

    def __init__(self, colony_memory: VolumetricMycelium, channel_groups,
                 context_scales=(1, 2), context_feature_channels=None):
        if not isinstance(colony_memory, VolumetricMycelium) or not colony_memory.centers.size:
            raise ValueError('a fitted VolumetricMycelium is required')
        groups = tuple(tuple(int(c) for c in group) for group in channel_groups)
        raw_scales = tuple(context_scales)
        if (not raw_scales or any(isinstance(scale, bool) or not isinstance(scale, int) or scale < 1
                                  for scale in raw_scales)):
            raise ValueError('context scales must be unique positive integers')
        scales = tuple(raw_scales)
        channels = colony_memory.centers.shape[1]
        flattened = [c for group in groups for c in group]
        if (not groups or any(not group or len(group) == channels for group in groups) or
                len(flattened) != len(set(flattened)) or
                any(c < 0 or c >= channels for c in flattened)):
            raise ValueError('channel groups must be nonempty, disjoint, and in range')
        if len(scales) != len(set(scales)):
            raise ValueError('context scales must be unique positive integers')
        self.colony_memory = colony_memory
        self.channel_groups = groups
        self.context_scales = tuple(sorted(scales))
        self.channels = channels
        self.colonies = colony_memory.colony_count
        if context_feature_channels is None:
            selected_context = tuple(range(channels))
        else:
            selected_context = tuple(int(c) for c in context_feature_channels)
            if (not selected_context or len(set(selected_context)) != len(selected_context) or
                    any(c < 0 or c >= channels for c in selected_context)):
                raise ValueError('context feature channels must be unique channel indices')
        self.context_feature_channels = selected_context
        self.ridge = colony_memory.config.predictive_ridge
        self.support = np.zeros((len(groups), self.colonies), dtype=np.int64)
        self.global_coefficients = []
        self.local_coefficients = []
        self.fitted = False

    @staticmethod
    def context_matrix(volume, hidden_channels, context_scales=(1, 2),
                       context_feature_channels=None):
        """Multi-radius 3D neighbors plus visible features at the center."""
        channels = volume.shape[-1]
        hidden = set(hidden_channels)
        selected = (tuple(range(channels)) if context_feature_channels is None else
                    tuple(context_feature_channels))
        if (not selected or len(set(selected)) != len(selected) or
                any(index < 0 or index >= channels for index in selected)):
            raise ValueError('context feature channels must be unique channel indices')
        neighbors = []
        for scale in context_scales:
            for axis in range(3):
                for delta in (-scale, scale):
                    shifted = np.zeros_like(volume)
                    dst = [slice(None)] * 4
                    src = [slice(None)] * 4
                    if delta > 0:
                        dst[axis], src[axis] = slice(0, -delta), slice(delta, None)
                    else:
                        dst[axis], src[axis] = slice(-delta, None), slice(0, delta)
                    shifted[tuple(dst)] = volume[tuple(src)]
                    neighbors.append(shifted.reshape(-1, channels)[:, selected])
        center = volume.reshape(-1, channels)
        visible_indices = [i for i in selected if i not in hidden]
        pieces = [*neighbors, center[:, visible_indices]]
        matrix = np.column_stack(pieces)
        return np.column_stack((matrix, np.ones(len(matrix), dtype=float)))

    def masked_examples(self, volume, group_index):
        """Return label-free context, target features, and masked colony IDs."""
        volume = self.colony_memory._validate_volume(volume)
        group = self.channel_groups[group_index]
        cues = volume.copy()
        cues[..., list(group)] = self.colony_memory.training_mean[list(group)]
        assignments = self.colony_memory.assignments(cues).reshape(-1)
        visible = [index for index in range(self.channels) if index not in group]
        active = np.max(np.abs(volume[..., visible]), axis=-1).reshape(-1) > self.colony_memory.config.activity_threshold
        valid = active
        features = self.context_matrix(volume, group, self.context_scales,
                                       self.context_feature_channels)
        targets = volume.reshape(-1, self.channels)[:, list(group)]
        return features[valid], targets[valid], assignments[valid]

    def fit(self, volumes):
        volumes = [self.colony_memory._validate_volume(v) for v in volumes]
        if not volumes:
            raise ValueError('at least one training volume is required')
        groups = len(self.channel_groups)
        accumulators = []
        target_counts = []
        for group in self.channel_groups:
            visible_context_count = sum(index not in group for index in self.context_feature_channels)
            feature_count = (6*len(self.context_scales)*len(self.context_feature_channels) +
                             visible_context_count + 1)
            output_count = len(group)
            global_xx = np.zeros((feature_count, feature_count), dtype=float)
            global_xy = np.zeros((feature_count, output_count), dtype=float)
            local_xx = np.zeros((self.colonies, feature_count, feature_count), dtype=float)
            local_xy = np.zeros((self.colonies, feature_count, output_count), dtype=float)
            support = np.zeros(self.colonies, dtype=np.int64)
            accumulators.append((global_xx, global_xy, local_xx, local_xy))
            target_counts.append(support)
        for volume in volumes:
            for group_index in range(groups):
                features, targets, assignments = self.masked_examples(volume, group_index)
                global_xx, global_xy, local_xx, local_xy = accumulators[group_index]
                global_xx += features.T @ features
                global_xy += features.T @ targets
                valid = assignments >= 0
                for colony in np.unique(assignments[valid]):
                    selected = assignments == colony
                    x, y = features[selected], targets[selected]
                    local_xx[colony] += x.T @ x
                    local_xy[colony] += x.T @ y
                    target_counts[group_index][colony] += int(selected.sum())
        for group_index, group in enumerate(self.channel_groups):
            global_xx, global_xy, local_xx, local_xy = accumulators[group_index]
            feature_count = global_xx.shape[0]
            penalty = np.eye(feature_count)*self.ridge
            penalty[-1, -1] = 0.0
            global_coef = np.linalg.solve(global_xx+penalty, global_xy)
            local_coef = np.empty((self.colonies, feature_count, len(group)), dtype=float)
            for colony in range(self.colonies):
                count = int(target_counts[group_index][colony])
                if count == 0:
                    local_coef[colony] = global_coef
                else:
                    fitted = np.linalg.solve(local_xx[colony]+penalty, local_xy[colony])
                    confidence = count/(count+64.0)
                    local_coef[colony] = confidence*fitted+(1.0-confidence)*global_coef
            self.global_coefficients.append(global_coef)
            self.local_coefficients.append(local_coef)
        self.support = np.stack(target_counts)
        self.fitted = True
        return self

    def predict_volume(self, volume, *, local_weights=1.0):
        if not self.fitted:
            raise RuntimeError('fit grouped predictive memory before inference')
        volume = self.colony_memory._validate_volume(volume)
        raw_weights = np.asarray(local_weights)
        if isinstance(local_weights, (bool, np.bool_)) or raw_weights.dtype.kind == 'b':
            raise ValueError('local_weights must be numeric probabilities, not booleans')
        weights = np.asarray(local_weights, dtype=float)
        if weights.ndim == 0:
            weights = np.full((len(self.channel_groups), self.colonies), float(weights))
        elif weights.shape == (len(self.channel_groups),):
            weights = np.repeat(weights[:, None], self.colonies, axis=1)
        elif weights.shape == (len(self.channel_groups), self.colony_memory.config.max_colonies):
            # Calibration may use the configured capacity while this fitted
            # memory instantiated fewer colonies. Only instantiated IDs can
            # be assigned, so ignore the unused capacity columns.
            weights = weights[:, :self.colonies]
        if (weights.shape != (len(self.channel_groups), self.colonies) or not np.all(np.isfinite(weights)) or
                np.any((weights < 0) | (weights > 1))):
            raise ValueError('local_weights must be one finite [0,1] value per group or colony')
        predictions = np.full(volume.shape, np.nan, dtype=float)
        local_predictions = np.full(volume.shape, np.nan, dtype=float)
        global_predictions = np.full(volume.shape, np.nan, dtype=float)
        colony_assignments = np.full((*volume.shape[:3], len(self.channel_groups)), -1, dtype=int)
        active_by_group = []
        for group_index, group in enumerate(self.channel_groups):
            visible = [index for index in range(self.channels) if index not in group]
            active = np.max(np.abs(volume[..., visible]), axis=-1).reshape(-1) > self.colony_memory.config.activity_threshold
            active_by_group.append(int(active.sum()))
            cue_volume = volume.copy()
            cue_volume[..., list(group)] = self.colony_memory.training_mean[list(group)]
            assignments = self.colony_memory.assignments(cue_volume).reshape(-1)
            features = self.context_matrix(volume, group, self.context_scales,
                                            self.context_feature_channels)[active]
            ids = assignments[active]
            colony_assignments[..., group_index] = assignments.reshape(volume.shape[:3])
            shared = features @ self.global_coefficients[group_index]
            specialist = shared.copy()
            for colony in np.unique(ids[ids >= 0]):
                selected = ids == colony
                specialist[selected] = features[selected] @ self.local_coefficients[group_index][colony]
            row_weights = np.zeros(len(ids), dtype=float)
            assigned = (ids >= 0) & (ids < self.colonies)
            row_weights[assigned] = weights[group_index, ids[assigned]]
            combined = ((1.0-row_weights[:, None])*shared +
                        row_weights[:, None]*specialist)
            positions = np.flatnonzero(active)
            for destination, values in ((global_predictions, shared),
                                        (local_predictions, specialist),
                                        (predictions, combined)):
                flat = destination.reshape(-1, self.channels)
                flat[positions[:, None], np.asarray(group)[None, :]] = values
        return {'predictions': predictions,
                'local_predictions': local_predictions,
                'global_predictions': global_predictions,
                'colony_assignments': colony_assignments,
                'active_patches_by_group': active_by_group,
                'context_scales': self.context_scales,
                'group_support': self.support.copy()}
