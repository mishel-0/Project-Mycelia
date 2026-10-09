"""Local directional context across ordered image slices.

This module reuses MYCELIA's unsupervised patch colonies, then learns sparse
six-direction transitions between colony IDs. The transition table is a
computational model inspired by local transport routes; it is not a claim that
fungi process images or interpret MRI signals biologically.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


# Each tuple is the offset from a source patch to the target patch. Slice
# neighbors are only connected when their numeric IDs differ by exactly one.
_NEIGHBORS = (
    (-1, 0, 0, 0),  # source on previous slice -> target, +z
    (1, 0, 0, 1),   # source on next slice -> target, -z
    (0, -1, 0, 2),  # source above -> target, +y
    (0, 1, 0, 3),   # source below -> target, -y
    (0, 0, -1, 4),  # source left -> target, +x
    (0, 0, 1, 5),  # source right -> target, -x
)


def blend_predictions(route, local, route_weight):
    """Mix learned route reconstruction with visible local image evidence."""
    if (isinstance(route_weight, bool) or not isinstance(route_weight, (int, float)) or
            not np.isfinite(route_weight) or not 0 <= route_weight <= 1):
        raise ValueError('route_weight must be finite and in [0, 1]')
    local = np.asarray(local, dtype=float)
    route = np.asarray(local if route is None else route, dtype=float)
    if route.shape != local.shape or not np.all(np.isfinite(route)) or not np.all(np.isfinite(local)):
        raise ValueError('route and local predictions must have matching finite shapes')
    return route_weight * route + (1 - route_weight) * local


class DirectionalVolumeContext:
    """Learn/predict a masked patch from local colony transitions in 2.5D.

    Inputs are per-patient mappings from ordered slice ID to a 2D array of
    colony assignments. Patient grouping is the caller's responsibility; fit
    must receive training patients only.
    """

    def __init__(self):
        self.routes = defaultdict(lambda: defaultdict(float))
        self.updates = 0
        self.patient_count = 0

    @staticmethod
    def _validate_volumes(volumes):
        normalized = {}
        for patient, slices in volumes.items():
            if not isinstance(slices, dict) or not slices:
                raise ValueError('each patient must map slice IDs to assignment grids')
            normalized[patient] = {}
            shape = None
            for slice_id, assignments in slices.items():
                if isinstance(slice_id, bool) or not isinstance(slice_id, int):
                    raise ValueError('slice IDs must be integers')
                grid = np.asarray(assignments)
                if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
                    raise ValueError('assignment grids must be 2D integer arrays')
                if np.any(grid < -1):
                    raise ValueError('assignments must be colony IDs or -1')
                if shape is None:
                    shape = grid.shape
                elif grid.shape != shape:
                    raise ValueError('all grids in a patient must have identical shape')
                normalized[patient][slice_id] = grid
        return normalized

    def fit(self, volumes):
        """Accumulate directed local routes, without labels or held-out patients."""
        volumes = self._validate_volumes(volumes)
        self.routes.clear()
        self.updates = 0
        self.patient_count = len(volumes)
        for slices in volumes.values():
            slice_ids = set(slices)
            for z, grid in slices.items():
                height, width = grid.shape
                for y in range(height):
                    for x in range(width):
                        target = int(grid[y, x])
                        if target < 0:
                            continue
                        for dz, dy, dx, direction in _NEIGHBORS:
                            source_slice = z + dz
                            if source_slice not in slice_ids:
                                continue
                            sy, sx = y + dy, x + dx
                            other = slices[source_slice]
                            if not 0 <= sy < height or not 0 <= sx < width:
                                continue
                            source = int(other[sy, sx])
                            if source < 0:
                                continue
                            self.routes[(source, direction)][target] += 1.0
                            self.updates += 1
        return self

    @property
    def route_count(self):
        return len(self.routes)

    def predict(self, volumes, patient, slice_id, y, x, model):
        """Predict target RGB patch from available six-connected neighbors.

        The target assignment is ignored even if the caller accidentally
        supplies one. Returned prototype pixels are MYCELIA's learned colony
        receptor traces, not retrieved image templates.
        """
        volumes = self._validate_volumes(volumes)
        if patient not in volumes or slice_id not in volumes[patient]:
            raise ValueError('unknown patient or slice')
        grid = volumes[patient][slice_id]
        if (isinstance(y, bool) or isinstance(x, bool) or
                not isinstance(y, int) or not isinstance(x, int) or
                not 0 <= y < grid.shape[0] or not 0 <= x < grid.shape[1]):
            raise ValueError('target coordinates outside assignment grid')
        votes = defaultdict(float)
        sources = 0
        slices = volumes[patient]
        slice_ids = set(slices)
        for dz, dy, dx, direction in _NEIGHBORS:
            neighbor_slice = slice_id + dz
            if neighbor_slice not in slice_ids:
                continue
            sy, sx = y + dy, x + dx
            if not 0 <= sy < grid.shape[0] or not 0 <= sx < grid.shape[1]:
                continue
            source = int(slices[neighbor_slice][sy, sx])
            if source < 0:
                continue
            row = self.routes.get((source, direction), {})
            total = sum(row.values())
            if total <= 0:
                continue
            sources += 1
            for target, count in row.items():
                votes[target] += count / total
        if not votes:
            return {'predicted_patch': None, 'context_sources': 0,
                    'candidate_probabilities': {}}

        total_votes = sum(votes.values())
        probabilities = {target: value / total_votes for target, value in votes.items()}
        patch_size = model.config.patch_size
        prediction = np.zeros((patch_size, patch_size, 3), dtype=float)
        for target, probability in probabilities.items():
            colony = model.colonies[target]
            pixels = [
                colony.organism.nodes[i].receptor_trace
                if colony.organism.nodes[i].alive and colony.organism.nodes[i].receptor_trace
                else [.5, .5, .5]
                for i in range(patch_size * patch_size)
            ]
            prediction += probability * np.asarray(pixels).reshape(patch_size, patch_size, 3)
        return {'predicted_patch': prediction, 'context_sources': sources,
                'candidate_probabilities': probabilities}
