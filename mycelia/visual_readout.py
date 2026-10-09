"""Label-free colony-match features for a separate research readout."""
from __future__ import annotations

import numpy as np

FEATURE_SCHEMA = 'mycelia-colony-match-features.v1'
SPATIAL_PYRAMID_SCHEMA = 'mycelia-spatial-pyramid-colony-features.v1'


def colony_match_features(model, cues, *, batch_size=128):
    """Summarize unsupervised patch-to-colony competition for each image.

    The vector contains four coarse spatial histograms of winning colony IDs,
    followed by each patch's three nearest colony mismatch values. It contains
    no class labels, paths, or identifiers. Its width is ``4*K + 48`` for K
    colonies in the frozen model.
    """
    cues = np.asarray(cues, dtype=float)
    side = model.config.size // model.config.patch_size
    if (cues.ndim != 4 or cues.shape[1:] !=
            (model.config.size, model.config.size, 3) or
            not np.all(np.isfinite(cues)) or np.any((cues < 0) | (cues > 1))):
        raise ValueError('invalid cue batch')
    if side != 4:
        raise ValueError('colony-match feature v1 requires a 4x4 patch grid')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    colony_count = model.colony_count
    result = np.zeros((len(cues), 4*colony_count + 16*3), dtype=np.float32)
    for start in range(0, len(cues), batch_size):
        end = min(start+batch_size, len(cues))
        score_batch = model.patch_scores_batch(cues[start:end])
        for local_index, scores in enumerate(score_batch):
            row = start + local_index
            histogram = result[row, :4*colony_count].reshape(4, colony_count)
            distances = result[row, 4*colony_count:].reshape(16, 3)
            for patch_index, colony_scores in enumerate(scores):
                finite = np.flatnonzero(np.isfinite(colony_scores))
                if not len(finite):
                    distances[patch_index] = 1.0
                    continue
                order = finite[np.argsort(colony_scores[finite], kind='stable')]
                winners = order[:3]
                values = colony_scores[winners]
                distances[patch_index, :len(values)] = values
                if len(values) < 3:
                    distances[patch_index, len(values):] = values[-1]
                py, px = divmod(patch_index, 4)
                quadrant = (py//2)*2 + px//2
                histogram[quadrant, int(order[0])] += .25
    if not np.all(np.isfinite(result)):
        raise ArithmeticError('nonfinite colony-match feature')
    return result


def centroid_distance_scores(model, features):
    """Return standardized squared distances to frozen readout centroids."""
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model['mean'], dtype=float)
    scale = np.asarray(model['scale'], dtype=float)
    centroids = np.asarray(model['centroids'], dtype=float)
    if (x.ndim != 2 or x.shape[1] != mean.size or scale.shape != mean.shape or
            centroids.ndim != 2 or centroids.shape[1] != mean.size or
            not np.all(np.isfinite(x))):
        raise ValueError('readout feature dimension or values are invalid')
    standardized = (x-mean)/scale
    return np.sum((standardized[:, None, :] - centroids[None, :, :])**2, axis=2)


def spatial_pyramid_features(model, cues, *, levels=(1, 2, 4), top_k=3,
                             batch_size=64):
    """Preserve colony winner locations at multiple spatial resolutions."""
    cues = np.asarray(cues, dtype=float)
    side = model.config.size // model.config.patch_size
    if (cues.ndim != 4 or cues.shape[1:] !=
            (model.config.size, model.config.size, 3) or
            not np.all(np.isfinite(cues)) or np.any((cues < 0) | (cues > 1))):
        raise ValueError('invalid cue batch')
    if (not isinstance(levels, tuple) or not levels or
            any(isinstance(level, bool) or not isinstance(level, int) or
                level < 1 or side % level for level in levels)):
        raise ValueError('levels must be a nonempty tuple of patch-grid divisors')
    if (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('top_k and batch_size must be positive integers')
    colony_count = model.colony_count
    patch_count = side*side
    histogram_width = sum(level*level for level in levels)*colony_count
    result = np.zeros((len(cues), histogram_width+patch_count*top_k), dtype=np.float32)
    for start in range(0, len(cues), batch_size):
        score_batch = model.patch_scores_batch(cues[start:start+batch_size])
        for local, scores in enumerate(score_batch):
            row = result[start+local]
            finite_scores = np.where(np.isfinite(scores), scores, np.inf)
            winners = np.argmin(finite_scores, axis=1)
            supported = np.isfinite(np.min(finite_scores, axis=1))
            offset = 0
            for level in levels:
                cell_side = side//level
                width = level*level*colony_count
                histogram = row[offset:offset+width].reshape(level*level, colony_count)
                for patch_index in np.flatnonzero(supported):
                    py, px = divmod(int(patch_index), side)
                    cell = (py//cell_side)*level + px//cell_side
                    histogram[cell, int(winners[patch_index])] += 1/(cell_side*cell_side)
                offset += width
            distances = row[histogram_width:].reshape(patch_count, top_k)
            for patch_index, colony_scores in enumerate(scores):
                finite = np.flatnonzero(np.isfinite(colony_scores))
                if not len(finite):
                    distances[patch_index] = 1.0
                    continue
                order = finite[np.argsort(colony_scores[finite], kind='stable')[:top_k]]
                values = colony_scores[order]
                distances[patch_index, :len(values)] = values
                if len(values) < top_k:
                    distances[patch_index, len(values):] = values[-1]
    if not np.all(np.isfinite(result)):
        raise ArithmeticError('nonfinite spatial-pyramid feature')
    return result
