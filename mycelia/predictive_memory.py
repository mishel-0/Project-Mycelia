"""Local masked-patch prediction using resource-funded graph memory.

This is a computational learning hypothesis. Each memory unit stores a local
neighbor-context trace and the patch that followed it in two real, pre-inoculated
compartment/cord scaffolds. It is not a growing hyphal network yet.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .memory import MemoryColony, MemoryConfig


@dataclass(frozen=True)
class PredictiveConfig:
    size: int = 32
    patch_size: int = 8
    max_units: int = 128
    novelty_threshold: float = .025
    correction_threshold: float = .004
    learning_rate: float = .04

    def __post_init__(self):
        for name in ('size', 'patch_size', 'max_units'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'invalid {name}')
        if self.patch_size < 3 or self.size % self.patch_size:
            raise ValueError('size must be divisible by patch_size >= 3')
        for name in ('novelty_threshold', 'correction_threshold', 'learning_rate'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid {name}')
        if self.correction_threshold >= self.novelty_threshold or self.learning_rate > 1:
            raise ValueError('invalid correction threshold or learning rate')


def patch_grid(cue, size=32, patch_size=8):
    cue = np.asarray(cue, dtype=float)
    if cue.shape != (size, size, 3) or not np.all(np.isfinite(cue)) or np.any((cue < 0) | (cue > 1)):
        raise ValueError('cue must be a finite size x size x 3 array in [0,1]')
    side = size // patch_size
    return cue.reshape(side, patch_size, side, patch_size, 3).transpose(0, 2, 1, 3, 4).reshape(side * side, patch_size, patch_size, 3)


def masked_neighbor_context(cue, target_index, *, size=32, patch_size=8):
    """Make an oriented context from visible four-neighbor patches only.

    Context channels encode neighbor mean intensity, left-to-right contrast,
    and top-to-bottom contrast. The target patch is explicitly hidden before
    the context is formed; changing its pixels cannot change this context.
    """
    patches = patch_grid(cue, size, patch_size)
    side = size // patch_size
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < side * side:
        raise ValueError('target_index is outside the patch grid')
    row, col = divmod(target_index, side)
    neighbors = {}
    if col > 0:
        neighbors['left'] = patches[target_index - 1][..., 0]
    if col + 1 < side:
        neighbors['right'] = patches[target_index + 1][..., 0]
    if row > 0:
        neighbors['top'] = patches[target_index - side][..., 0]
    if row + 1 < side:
        neighbors['bottom'] = patches[target_index + side][..., 0]
    available = list(neighbors.values())
    if not available:
        raise ValueError('target patch has no visible neighbors')
    mean = np.mean(available, axis=0)
    left = neighbors.get('left', mean)
    right = neighbors.get('right', mean)
    top = neighbors.get('top', mean)
    bottom = neighbors.get('bottom', mean)
    dx = .5 + .5 * np.clip(left - right, -1, 1)
    dy = .5 + .5 * np.clip(top - bottom, -1, 1)
    return np.stack((mean, dx, dy), axis=-1)


def _scaffold_config(config):
    return MemoryConfig(size=config.patch_size, adaptive=True, fast_rate=0,
                        slow_rate=config.learning_rate, physiology_interval=20)


class _PredictiveUnit:
    def __init__(self, memory_config, position=None):
        self.context = MemoryColony(memory_config)
        self.target = MemoryColony(memory_config)
        self.exposures = 0
        self.position = position

    def state_dict(self):
        return dict(context=self.context.state_dict(), target=self.target.state_dict(), exposures=self.exposures,
                    position=self.position)

    @classmethod
    def from_state(cls, data):
        obj = cls.__new__(cls)
        obj.context = MemoryColony.from_state(data['context'])
        obj.target = MemoryColony.from_state(data['target'])
        obj.exposures = data['exposures']
        obj.position = data['position']
        return obj


class PredictiveMycelium:
    """Competitive context-to-patch memory trained by prediction residuals."""
    schema = 'mycelia-masked-predictive-1'

    def __init__(self, config=None):
        self.config = config or PredictiveConfig()
        self.memory_config = _scaffold_config(self.config)
        self.units = []
        self.frozen = False
        self.images_seen = 0
        self.samples_seen = 0
        self.updates = 0
        self.suppressed_updates = 0
        self.novelty_overflow = 0
        # The cache is reconstructed only from living graph receptors and cords.
        self._cache_dirty = True
        self._context_traces = None
        self._context_routes = None
        self._context_weights = None
        self._route_pairs = None

    def _validate_cue(self, cue):
        value = np.asarray(cue, dtype=float)
        expected = (self.config.patch_size, self.config.patch_size, 3)
        if value.shape != expected or not np.all(np.isfinite(value)) or np.any((value < 0) | (value > 1)):
            raise ValueError(f'cue must have shape {expected} and values in [0,1]')
        return value

    @staticmethod
    def _stored_patch(colony):
        s = colony.config.size
        return np.asarray([colony.organism.nodes[i].receptor_trace or [.5] * 3 for i in range(s * s)], dtype=float).reshape(s, s, 3)

    def _refresh_cache(self):
        if not self._cache_dirty:
            return
        s = self.config.patch_size
        self._context_traces = np.asarray([self._stored_patch(unit.context) for unit in self.units]) if self.units else np.empty((0, s, s, 3))
        pairs = sorted({(edge.a, edge.b) for unit in self.units for edge in unit.context.organism.segments.values()
                        if edge.alive and unit.context.organism.nodes[edge.a].alive and unit.context.organism.nodes[edge.b].alive})
        self._route_pairs = np.asarray(pairs, dtype=int).reshape(-1, 2) if pairs else np.empty((0, 2), dtype=int)
        self._context_routes = np.zeros((len(self.units), len(pairs)))
        self._context_weights = np.zeros_like(self._context_routes)
        if pairs:
            for unit_index in range(len(self.units)):
                self._refresh_unit(unit_index)
        self._cache_dirty = False

    def _refresh_unit(self, index):
        unit = self.units[index]
        self._context_traces[index] = self._stored_patch(unit.context)
        if not len(self._route_pairs):
            return
        pair_index = {tuple(pair): i for i, pair in enumerate(self._route_pairs.tolist())}
        route = np.zeros(len(self._route_pairs))
        weights = np.zeros(len(self._route_pairs))
        organism = unit.context.organism
        for edge in organism.segments.values():
            if edge.alive and organism.nodes[edge.a].alive and organism.nodes[edge.b].alive:
                i = pair_index[(edge.a, edge.b)]
                route[i] = edge.route_trace
                weights[i] = edge.conductance(organism.config.hydraulic_scale)
        weights /= max(float(weights.sum()), 1e-12)
        self._context_routes[index] = route
        self._context_weights[index] = weights

    def _position(self, position):
        if position is None:
            return None
        count = (self.config.size // self.config.patch_size) ** 2
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < count:
            raise ValueError('invalid target patch position')
        return position

    def _scores(self, context, position=None):
        self._refresh_cache()
        if not self.units:
            return np.empty(0)
        x = self._validate_cue(context)
        delta = self._context_traces - x[None, ...]
        node_error = np.mean(delta * delta, axis=(1, 2, 3))
        route_error = np.zeros(len(self.units))
        if len(self._route_pairs):
            aa, bb = self._route_pairs.T
            flat = x.reshape(-1, 3)
            contrast = np.abs(flat[aa] - flat[bb]).mean(axis=1)
            differences = contrast[None, :] - self._context_routes
            route_error = np.sum(self._context_weights * differences * differences, axis=1)
        scores = node_error + .15 * route_error
        position = self._position(position)
        if position is not None:
            eligible = np.asarray([unit.position == position for unit in self.units])
            scores = scores[eligible]
        return scores

    def predict(self, context, position=None):
        """Predict a hidden patch from a context cue; never changes graph state."""
        x = self._validate_cue(context)
        position = self._position(position)
        scores = self._scores(x, position)
        if not len(scores):
            return np.full_like(x, .5), None, math.inf
        local_winner = int(np.argmin(scores))
        if position is None:
            winner = local_winner
        else:
            winner = [i for i, unit in enumerate(self.units) if unit.position == position][local_winner]
        return self._stored_patch(self.units[winner].target), winner, float(scores[local_winner])

    def learn(self, context, arriving_target, position=None):
        """Predict first, then adjust the winning local context/target connection."""
        if self.frozen:
            raise RuntimeError('predictive memory is frozen')
        context = self._validate_cue(context)
        target = self._validate_cue(arriving_target)
        position = self._position(position)
        before, winner, context_error = self.predict(context, position)
        prediction_error = float(np.mean((before[..., 0] - target[..., 0]) ** 2))
        novel = winner is None or context_error > self.config.novelty_threshold
        recruited = winner is None or (novel and len(self.units) < self.config.max_units)
        if winner is None and not recruited:
            self.novelty_overflow += 1
            self.suppressed_updates += 1
            self.samples_seen += 1
            return dict(prediction_mse=prediction_error, context_error=context_error,
                        winner=None, recruited=0, updated=0)
        if recruited:
            unit = _PredictiveUnit(self.memory_config, position)
            self.units.append(unit)
            winner = len(self.units) - 1
            self._cache_dirty = True
        elif novel:
            self.novelty_overflow += 1
        if prediction_error <= self.config.correction_threshold and not novel:
            self.suppressed_updates += 1
        else:
            reward = min(1., max(.1, math.sqrt(max(prediction_error, 1e-12) / .04)))
            unit = self.units[winner]
            unit.context._expose(context, reward=reward)
            unit.target._expose(target, reward=reward)
            unit.exposures += 1
            self.updates += 1
            if self._cache_dirty:
                self._refresh_cache()
            else:
                self._refresh_unit(winner)
        self.samples_seen += 1
        return dict(prediction_mse=prediction_error, context_error=context_error,
                    winner=winner, recruited=int(recruited),
                    updated=int(prediction_error > self.config.correction_threshold or novel))

    def freeze(self):
        self.frozen = True
        for unit in self.units:
            unit.context.freeze(); unit.target.freeze()

    def validate(self):
        if not isinstance(self.frozen, bool) or len(self.units) > self.config.max_units:
            raise ValueError('invalid predictive memory state')
        for value in (self.images_seen, self.samples_seen, self.updates, self.suppressed_updates, self.novelty_overflow):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('invalid predictive memory counter')
        if self.updates + self.suppressed_updates != self.samples_seen:
            raise ValueError('predictive update accounting mismatch')
        for unit in self.units:
            if unit.context.config != self.memory_config or unit.target.config != self.memory_config:
                raise ValueError('inconsistent predictive scaffold configuration')
            if unit.context.organism.memory_frozen != self.frozen or unit.target.organism.memory_frozen != self.frozen:
                raise ValueError('inconsistent predictive freeze state')
            self._position(unit.position)
            if isinstance(unit.exposures, bool) or not isinstance(unit.exposures, int) or unit.exposures < 0:
                raise ValueError('invalid predictive unit exposure count')
            if unit.context.exposures != unit.exposures or unit.target.exposures != unit.exposures:
                raise ValueError('context/target exposure pairing mismatch')
            unit.context.organism._validate(); unit.target.organism._validate()
        if sum(unit.exposures for unit in self.units) != self.updates:
            raise ValueError('predictive unit update accounting mismatch')

    def summary(self):
        self.validate()
        budgets = [abs(unit.context.organism.summary()['carbon_error']) for unit in self.units]
        budgets.extend(abs(unit.target.organism.summary()['carbon_error']) for unit in self.units)
        return dict(images_seen=self.images_seen, samples_seen=self.samples_seen,
                    predictive_units=len(self.units), scaffold_count=2 * len(self.units),
                    updates=self.updates, suppressed_updates=self.suppressed_updates,
                    novelty_overflow=self.novelty_overflow,
                    context_exposures=sum(unit.context.exposures for unit in self.units),
                    target_exposures=sum(unit.target.exposures for unit in self.units),
                    max_absolute_budget_error=max(budgets, default=0.))

    def state_dict(self):
        self.validate()
        return dict(schema=self.schema, config=asdict(self.config), frozen=self.frozen,
                    images_seen=self.images_seen, samples_seen=self.samples_seen, updates=self.updates,
                    suppressed_updates=self.suppressed_updates, novelty_overflow=self.novelty_overflow,
                    units=[unit.state_dict() for unit in self.units])

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(self.state_dict(), allow_nan=False))
        temp.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('schema') != cls.schema:
            raise ValueError('unsupported predictive memory schema')
        model = cls(PredictiveConfig(**data['config']))
        for name in ('frozen', 'images_seen', 'samples_seen', 'updates', 'suppressed_updates', 'novelty_overflow'):
            setattr(model, name, data[name])
        model.units = [_PredictiveUnit.from_state(unit) for unit in data['units']]
        model.validate()
        return model
