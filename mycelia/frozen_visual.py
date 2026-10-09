"""Compact inference snapshot of a frozen MYCELIA graph, with identical scores.

This is an execution optimization, not a new learner or biological mechanism.
The full checkpoint remains necessary for inspection, reconstruction, and any
future learning. Only state used by the current colony-match readout is copied.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .visual_memory import (VisualConfig, VisualMycelium, _graph_score_arrays,
                            _scores_from_arrays)

SCHEMA = 'mycelia.frozen-graph-matcher.v1'
ARRAY_KEYS = {'traces', 'node_weights', 'weights', 'routes',
              'edge_a', 'edge_b', 'supported'}


class FrozenVisualMatcher:
    """An independent, immutable numeric snapshot for repeated inference."""

    def __init__(self, config, arrays, source_memory_sha256):
        if (not isinstance(source_memory_sha256, str) or
                len(source_memory_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in source_memory_sha256)):
            raise ValueError('a source-memory SHA-256 is required')
        self.config = config
        self.source_memory_sha256 = source_memory_sha256
        self.frozen = True
        if set(arrays) != ARRAY_KEYS:
            raise ValueError('invalid frozen graph array keys')
        self._arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
        self._validate()
        for value in self._arrays.values():
            value.setflags(write=False)

    @classmethod
    def from_model(cls, model, source_memory_sha256):
        if not model.frozen:
            raise ValueError('only a frozen MYCELIA model can be compiled')
        model.validate()
        return cls(model.config, _graph_score_arrays(model.colonies), source_memory_sha256)

    @property
    def colony_count(self):
        return self._arrays['traces'].shape[0]

    @property
    def array_bytes(self):
        return sum(value.nbytes for value in self._arrays.values())

    def _validate(self):
        a = self._arrays
        n = self.config.patch_size ** 2
        traces = a['traces']
        if traces.ndim != 2 or traces.shape[0] < 1 or traces.shape[1] != n*3:
            raise ValueError('invalid trace dimensions')
        count = traces.shape[0]
        edges = a['edge_a']
        if edges.ndim != 1 or a['edge_b'].shape != edges.shape:
            raise ValueError('invalid edge dimensions')
        if (a['node_weights'].shape != traces.shape or
                a['weights'].shape != (count, len(edges)) or
                a['routes'].shape != a['weights'].shape or
                a['supported'].shape != (count,) or
                a['supported'].dtype != np.bool_):
            raise ValueError('invalid frozen graph dimensions or support mask')
        for key in ('edge_a', 'edge_b'):
            values = a[key]
            if values.dtype.kind not in 'iu' or np.any((values < 0) | (values >= n)):
                raise ValueError('invalid edge endpoints')
        for key in ('traces', 'node_weights', 'weights', 'routes'):
            values = a[key]
            if values.dtype != np.float64 or not np.all(np.isfinite(values)):
                raise ValueError('frozen graph requires finite float64 arrays')
        if np.any(a['node_weights'] < 0) or np.any(a['weights'] < 0):
            raise ValueError('negative graph evidence weights')

    def encode_image(self, image):
        return VisualMycelium.encode_image(self, image)

    def patches(self, cue):
        return VisualMycelium.patches(self, cue)

    def patch_scores(self, cue):
        return _scores_from_arrays(self._arrays, self.patches(cue))

    def patch_scores_batch(self, cues):
        cues = np.asarray(cues, dtype=float)
        c = self.config
        if (cues.ndim != 4 or cues.shape[1:] != (c.size, c.size, 3) or
                not np.all(np.isfinite(cues)) or np.any((cues < 0) | (cues > 1))):
            raise ValueError('invalid visual cue batch')
        count = (c.size // c.patch_size)**2
        if not len(cues):
            return np.empty((0, count, self.colony_count))
        patches = np.concatenate([self.patches(cue) for cue in cues], axis=0)
        return _scores_from_arrays(self._arrays, patches).reshape(len(cues), count, -1)

    def save(self, path):
        path = Path(path)
        metadata = json.dumps({'schema': SCHEMA, 'config': asdict(self.config),
                               'source_memory_sha256': self.source_memory_sha256})
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, metadata=np.asarray(metadata), **self._arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != ARRAY_KEYS | {'metadata'}:
                raise ValueError('invalid frozen graph archive')
            metadata = json.loads(str(data['metadata']))
            if metadata.get('schema') != SCHEMA:
                raise ValueError('unsupported frozen graph schema')
            return cls(VisualConfig(**metadata['config']),
                       {key: data[key] for key in ARRAY_KEYS},
                       metadata['source_memory_sha256'])
