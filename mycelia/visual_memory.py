"""Label-free local visual memory, with resource-paid graph imprints.

Fixed image preprocessing and competitive vector quantization are computational
hypotheses, not measured fungal image-learning laws. No learned encoder, neural
network, class labels, or external fitted readout is used. New lattices are funded
inocula, not claims of simulated branching from a parent hypha.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import base64
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import zlib

import numpy as np

from .memory import MemoryColony, MemoryConfig, image_cues


@dataclass(frozen=True)
class VisualConfig:
    size: int = 32
    patch_size: int = 8
    max_colonies: int = 128
    max_episodes: int = 10_000
    recognition_threshold: float = .08
    novelty_threshold: float = .012
    # Disabled by default until the matched validation ablation is accepted.
    # At capacity, sufficiently similar patch specialists may fuse into one
    # weighted motif, releasing one funded slot for an unfamiliar pattern.
    motif_merge_threshold: float = 0.0
    correction_threshold: float = .001
    predictive_error_scale: float = .01
    predictive_feedback: bool = False
    learning_rate: float = .04
    detailed: bool = True
    multiscale_features: bool = False
    preprocessing: str = 'foreground'
    hierarchical_context: bool = False
    cue_mode: str = 'edge_texture'
    specialization: bool = True
    error_gated: bool = True

    def __post_init__(self):
        for key in ('size', 'patch_size', 'max_colonies', 'max_episodes'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'invalid {key}')
        if self.patch_size < 3 or self.size % self.patch_size:
            raise ValueError('size must be divisible by patch_size >= 3')
        for key in ('novelty_threshold', 'correction_threshold', 'predictive_error_scale', 'learning_rate', 'recognition_threshold'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid {key}')
        if self.correction_threshold >= self.novelty_threshold or self.learning_rate > 1:
            raise ValueError('invalid correction threshold or learning rate')
        if (isinstance(self.motif_merge_threshold, bool) or
                not isinstance(self.motif_merge_threshold, (int, float)) or
                not math.isfinite(self.motif_merge_threshold) or
                not 0 <= self.motif_merge_threshold <= .25):
            raise ValueError('motif_merge_threshold must be in [0, .25]')
        for key in ('detailed', 'multiscale_features', 'specialization', 'error_gated', 'predictive_feedback', 'hierarchical_context'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f'invalid {key}')
        if not isinstance(self.preprocessing, str) or self.preprocessing not in {'foreground', 'fixed'}:
            raise ValueError('preprocessing must be foreground or fixed')
        if self.cue_mode not in {'edge_texture', 'oriented', 'axis_edges'}:
            raise ValueError('cue_mode must be edge_texture, oriented, or axis_edges')


def prepare_gray(image, size=32):
    """Foreground bounding box, aspect-preserving resize, robust contrast.

    This is an intensity foreground crop, not anatomical skull stripping. Pillow
    is imported only for image operations; the graph core still needs only NumPy.
    """
    from PIL import Image
    if isinstance(size, bool) or not isinstance(size, int) or size < 3:
        raise ValueError('size must be an integer >= 3')
    if hasattr(image, 'convert'):
        gray = np.asarray(image.convert('L'), dtype=float) / 255
    else:
        gray = np.asarray(image, dtype=float)
    if gray.ndim != 2 or not gray.size or not np.all(np.isfinite(gray)) or np.any((gray < 0) | (gray > 1)):
        raise ValueError('image must be finite grayscale in [0,1]')
    foreground = gray > .02
    if foreground.any():
        yy, xx = np.where(foreground)
        gray = gray[yy.min():yy.max()+1, xx.min():xx.max()+1]
        values = gray[gray > .02]
        low, high = np.quantile(values, [.01, .99])
        if high - low > .05:
            gray = np.clip((gray - low) / (high - low), 0, 1)
    height, width = gray.shape
    ratio = min(size / height, size / width)
    w, h = max(1, round(width * ratio)), max(1, round(height * ratio))
    small = np.asarray(Image.fromarray(np.rint(gray * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BOX), dtype=float) / 255
    result = np.zeros((size, size), dtype=float)
    y, x = (size - h) // 2, (size - w) // 2
    result[y:y+h, x:x+w] = small
    return result


def prepare_gray_fixed(image, size=32):
    """Resize the complete image to a fixed canvas without image-wise statistics.

    Unlike :func:`prepare_gray`, this path does not crop to foreground or
    estimate contrast from the image. Each output pixel depends only on the
    corresponding fixed input footprint, so a later masked-patch benchmark
    cannot leak target pixels through a scan-wide crop or normalization step.
    Intensity values keep the source's [0, 1] grayscale scale.
    """
    from PIL import Image
    if isinstance(size, bool) or not isinstance(size, int) or size < 3:
        raise ValueError('size must be an integer >= 3')
    if hasattr(image, 'convert'):
        gray = np.asarray(image.convert('L'), dtype=float) / 255
    else:
        gray = np.asarray(image, dtype=float)
    if gray.ndim != 2 or not gray.size or not np.all(np.isfinite(gray)) or np.any((gray < 0) | (gray > 1)):
        raise ValueError('image must be finite grayscale in [0,1]')
    height, width = gray.shape
    ratio = min(size / height, size / width)
    w, h = max(1, round(width * ratio)), max(1, round(height * ratio))
    small = np.asarray(Image.fromarray(np.rint(gray * 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BOX), dtype=float) / 255
    result = np.zeros((size, size), dtype=float)
    y, x = (size - h) // 2, (size - w) // 2
    result[y:y+h, x:x+w] = small
    return result


def visual_cues(gray, detailed=True, multiscale=False, mode='edge_texture'):
    """Intensity, edge magnitude, and local texture; fixed, not learned.

    The optional multi-scale mode combines fine gradients/textures with broader
    9x9 and 13x13 neighborhoods while keeping the learner's three-channel
    receptor interface. It adds context to the cue, not a learned hierarchy.
    """
    gray = np.asarray(gray, dtype=float)
    # image_cues supplies the shared input validity checks.
    basic = image_cues(gray)
    if not isinstance(detailed, bool) or not isinstance(multiscale, bool):
        raise ValueError('detailed and multiscale must be boolean')
    if mode not in {'edge_texture', 'oriented', 'axis_edges'}:
        raise ValueError('mode must be edge_texture, oriented, or axis_edges')
    if not detailed:
        return basic
    gy, gx = np.gradient(gray)
    if mode in {'oriented', 'axis_edges'}:
        if multiscale:
            broad_pad = np.pad(gray, 6, mode='edge')
            broad_windows = np.lib.stride_tricks.sliding_window_view(broad_pad, (13, 13))
            broad_mean = broad_windows.mean(axis=(-2, -1))
            broad_gy, broad_gx = np.gradient(broad_mean)
            gx, gy = .5*(gx+broad_gx), .5*(gy+broad_gy)
        if mode == 'oriented':
            return np.stack((gray, np.clip(.5+2*gx, 0, 1),
                             np.clip(.5+2*gy, 0, 1)), axis=-1)
        return np.stack((gray, np.clip(4*np.abs(gx), 0, 1),
                         np.clip(4*np.abs(gy), 0, 1)), axis=-1)
    edge = np.clip(4*np.hypot(gx, gy), 0, 1)
    padded = np.pad(gray, 2, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    texture = np.clip(4*windows.std(axis=(-2, -1)), 0, 1)
    if multiscale:
        broad_pad = np.pad(gray, 6, mode='edge')
        broad_windows = np.lib.stride_tricks.sliding_window_view(broad_pad, (13, 13))
        broad_mean = broad_windows.mean(axis=(-2, -1))
        broad_texture = np.clip(4*broad_windows.std(axis=(-2, -1)), 0, 1)
        broad_gy, broad_gx = np.gradient(broad_mean)
        broad_edge = np.clip(4*np.hypot(broad_gx, broad_gy), 0, 1)
        edge = .5*(edge+broad_edge)
        texture = .5*(texture+broad_texture)
    return np.stack((gray, edge, texture), axis=-1)


def mask_gray_patch(gray, patch_index, patch_size=8):
    """Hide one grayscale patch before cue extraction using only visible pixels.

    This prevents local edge/texture windows outside the target from carrying
    its true values into a masked-region prediction benchmark.
    """
    gray = np.asarray(gray, dtype=float)
    if (gray.ndim != 2 or not gray.size or not np.all(np.isfinite(gray)) or
            np.any((gray < 0) | (gray > 1))):
        raise ValueError('gray must be finite and in [0,1]')
    if (isinstance(patch_size, bool) or not isinstance(patch_size, int) or
            patch_size < 1 or gray.shape[0] % patch_size or gray.shape[1] % patch_size):
        raise ValueError('patch_size must divide both image dimensions')
    grid_y, grid_x = gray.shape[0]//patch_size, gray.shape[1]//patch_size
    count = grid_y*grid_x
    if (isinstance(patch_index, bool) or not isinstance(patch_index, int) or
            not 0 <= patch_index < count):
        raise ValueError('patch_index is outside the image grid')
    py, px = divmod(patch_index, grid_x)
    y0, x0 = py*patch_size, px*patch_size
    visible = np.ones(gray.shape, dtype=bool)
    visible[y0:y0+patch_size, x0:x0+patch_size] = False
    if not visible.any():
        raise ValueError('cannot mask the entire image')
    result = gray.copy()
    result[y0:y0+patch_size, x0:x0+patch_size] = float(np.mean(gray[visible]))
    return result


def _graph_score_arrays(colonies):
    """Take a numeric snapshot of the current living graph's scoring state."""
    if not colonies:
        raise ValueError('cannot compile an empty colony population')
    s = colonies[0].config.size
    # All inocula share geometry. Retired edges/nodes contribute no evidence.
    pairs = sorted({(e.a, e.b) for c in colonies for e in c.organism.segments.values()
                    if e.alive and c.organism.nodes[e.a].alive and c.organism.nodes[e.b].alive})
    pi = {pair: i for i, pair in enumerate(pairs)}
    aa = np.array([a for a, b in pairs], dtype=int)
    bb = np.array([b for a, b in pairs], dtype=int)
    traces, node_weights = [], []
    weights = np.zeros((len(colonies), len(pairs)))
    routes = np.zeros_like(weights)
    supported = []
    for j, colony in enumerate(colonies):
        o = colony.organism
        traces.append([o.nodes[i].receptor_trace or [.5]*3 for i in range(s*s)])
        live = np.array([o.nodes[i].alive for i in range(s*s)], dtype=float)
        node_weights.append(np.repeat(live, 3) / max(3 * live.sum(), 1))
        supported.append(colony.has_trace())
        for e in o.segments.values():
            if e.alive and o.nodes[e.a].alive and o.nodes[e.b].alive:
                k = pi[(e.a, e.b)]
                weights[j, k] = e.conductance(o.config.hydraulic_scale)
                routes[j, k] = e.route_trace
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    traces = np.asarray(traces).reshape(len(colonies), -1)
    nw = np.asarray(node_weights)
    return dict(traces=traces, node_weights=nw, weights=weights, routes=routes,
                edge_a=aa, edge_b=bb, supported=np.asarray(supported, dtype=bool))


def _scores_from_arrays(state, patches):
    """Evaluate the same node/route metric without rebuilding graph objects."""
    traces, nw = state['traces'], state['node_weights']
    weights, routes = state['weights'], state['routes']
    aa, bb, supported = state['edge_a'], state['edge_b'], state['supported']
    s = patches.shape[1]
    flat = patches.reshape(len(patches), s*s, 3)
    x = flat.reshape(len(flat), -1)
    node_error = np.maximum(0, x*x @ nw.T + np.sum(nw*traces*traces, axis=1)[None, :] - 2*x @ (nw*traces).T)
    contrast = np.abs(flat[:, aa] - flat[:, bb]).mean(axis=2)
    route_error = np.maximum(0, contrast*contrast @ weights.T + np.sum(weights*routes*routes, axis=1)[None, :] - 2*contrast @ (weights*routes).T)
    result = node_error + .15 * route_error
    result[:, np.logical_not(supported)] = np.inf
    return result


def _distances(colonies, patches):
    """Rebuild scratch arrays from actual living graph state on every call."""
    if not colonies:
        return np.empty((len(patches), 0))
    return _scores_from_arrays(_graph_score_arrays(colonies), patches)


class VisualMycelium:
    """Unlabeled patch specialists; recruitment and correction follow mismatch.

    Error-aware competition means reconstruction residual feedback, not correcting
    tumor diagnoses. The learner accepts only cue arrays, never a category or ID.
    """
    def __init__(self, config=None):
        self.config = config or VisualConfig()
        self.colonies = []
        self.frozen = False
        self.images_seen = 0
        self.patches_seen = 0
        self.updates = 0
        self.suppressed_updates = 0
        self.novelty_overflow = 0
        self.recycled_colonies = 0
        self.selection_counts = []
        # Recent/current specialist support resets when a retired slot is
        # recycled. Unlike selection_counts, it is safe to use as a fusion
        # weight because it never includes a previous colony life.
        self.colony_support = []
        self.motif_merges = 0
        # Whole-image episodes complement local colony pattern memory. The
        # canonical template is the actual visual cue, not a class or filename.
        self.episodes = []
        self._episode_index = {}
        self._episode_template_cache = None
        self.next_episode_id = 0
        # Sparse, label-free predictive links between neighboring patch
        # specialists. Keys are (source colony, direction, target colony).
        # These remain separate from exact whole-image episodic recall.
        self.context_associations = {}
        self._context_row_totals = {}
        self._context_row_targets = {}
        self.context_updates = 0
        self.context_prediction_error = 0.0
        # Sparse three-patch motifs: two direction-tagged neighboring colony
        # identities predict a target colony. This higher-order memory is an
        # optional algorithmic hypothesis layered above one-edge routes.
        self.context_motifs = {}
        self._context_motif_row_totals = {}
        self._context_motif_row_targets = {}
        self.context_motif_updates = 0
        self.predictive_feedback_patches = 0
        self.predictive_feedback_error = 0.0
        # Each complete new scaffold draws its carbon, water, and energy from
        # explicit initial reserves. No free cloning or unrecorded inoculation.
        unit = MemoryColony(self._memory_config()).organism
        self.unit_funding = dict(carbon=unit.initial_carbon, water=unit.initial_water, energy=unit.initial_energy)
        self.initial_funding = {k: v*self.config.max_colonies for k, v in self.unit_funding.items()}
        self.remaining_funding = dict(self.initial_funding)

    def _memory_config(self):
        return MemoryConfig(size=self.config.patch_size, adaptive=True, fast_rate=0,
                            slow_rate=self.config.learning_rate, physiology_interval=20)

    @property
    def colony_count(self):
        return len(self.colonies)

    def encode_image(self, image):
        """Preprocess one grayscale/Pillow image with this model's saved cue mode."""
        preprocess = prepare_gray_fixed if self.config.preprocessing == 'fixed' else prepare_gray
        gray = preprocess(image, self.config.size)
        return visual_cues(gray, self.config.detailed, self.config.multiscale_features,
                           self.config.cue_mode)

    def learn_image(self, image):
        """Encode and learn one image without accepting labels or identifiers."""
        return self.learn(self.encode_image(image))

    def recognize_image(self, image):
        """Encode and perform cue-addressed recognition with this model's mode."""
        return self.recognize(self.encode_image(image))

    def reconstruct_image(self, image):
        """Encode and reconstruct an image from its stored colony cues."""
        return self.reconstruct(self.encode_image(image))

    def patches(self, cue):
        c = self.config
        cue = np.asarray(cue, dtype=float)
        if cue.shape != (c.size, c.size, 3) or not np.all(np.isfinite(cue)) or np.any((cue < 0) | (cue > 1)):
            raise ValueError('invalid visual cue')
        p, n = c.patch_size, c.size // c.patch_size
        return cue.reshape(n, p, n, p, 3).transpose(0, 2, 1, 3, 4).reshape(n*n, p, p, 3)

    def patch_scores(self, cue):
        return _distances(self.colonies, self.patches(cue))

    def patch_scores_batch(self, cues):
        """Return patch-to-colony mismatches for a batch of visual cues.

        The batch axis is preserved as ``(images, patches, colonies)``. This
        rebuilds the same scratch arrays used by :meth:`patch_scores` and does
        not alter learning state.
        """
        cues = np.asarray(cues, dtype=float)
        c = self.config
        if (cues.ndim != 4 or cues.shape[1:] != (c.size, c.size, 3) or
                not np.all(np.isfinite(cues)) or np.any((cues < 0) | (cues > 1))):
            raise ValueError('invalid visual cue batch')
        if not len(cues):
            return np.empty((0, (c.size//c.patch_size)**2, len(self.colonies)))
        patches = np.concatenate([self.patches(cue) for cue in cues], axis=0)
        scores = _distances(self.colonies, patches)
        return scores.reshape(len(cues), (c.size//c.patch_size)**2, len(self.colonies))

    def _recruit(self, patch):
        recycle_index = None
        if len(self.colonies) >= self.config.max_colonies:
            # A fully retired colony has already relinquished its living
            # scaffold. Reuse that slot's original funded allocation instead
            # of permanently losing growth capacity. Live specialists are
            # never evicted by this rule.
            recycle_index = next((i for i, colony in enumerate(self.colonies)
                                  if not any(node.alive for node in colony.organism.nodes.values())), None)
            if recycle_index is None:
                return None
        elif any(self.remaining_funding[k] + 1e-10*max(1., v) < v for k, v in self.unit_funding.items()):
            return None
        colony = MemoryColony(self._memory_config())
        # This unregistered inoculum is discarded if synthesis fails. The image
        # transaction owns rollback; a second full graph copy is unnecessary.
        result = colony._expose(patch)
        if result['synthesis_carbon'] <= 0:
            return None
        if recycle_index is not None:
            old = self.colonies[recycle_index]
            rollback = getattr(self, '_active_recruit_rollback', None)
            if rollback is not None and recycle_index not in rollback:
                rollback[recycle_index] = (deepcopy(old.organism.__dict__), old.exposures)
            colony.exposures += old.exposures
            self.colonies[recycle_index] = colony
            self.colony_support[recycle_index] = 0
            # Colony indices are reused, but learned routes belonged to the
            # retired tissue and must not be inherited by a new specialist.
            stale = [key for key in self.context_associations
                     if key[0] == recycle_index or key[2] == recycle_index]
            for key in stale:
                del self.context_associations[key]
            self._context_row_totals = {}
            self._context_row_targets = {}
            for (source, direction, target), weight in self.context_associations.items():
                row = (source, direction)
                self._context_row_totals[row] = self._context_row_totals.get(row, 0.0)+weight
                self._context_row_targets.setdefault(row, {})[target] = weight
            stale_motifs = [key for key in self.context_motifs
                            if key[0] == recycle_index or key[2] == recycle_index or
                               key[4] == recycle_index]
            for key in stale_motifs:
                del self.context_motifs[key]
            self._rebuild_context_motif_indices()
            result['colony_index'] = recycle_index
            result['recycled_slot'] = True
            return result
        for k, v in self.unit_funding.items():
            # Repeated floating-point subtraction can falsely deny the last
            # fully funded inoculum. Derive reserves from committed allocation.
            self.remaining_funding[k] = max(0., self.initial_funding[k]-(len(self.colonies)+1)*v)
        self.colonies.append(colony)
        self.selection_counts.append(0)
        self.colony_support.append(0)
        result['colony_index'] = len(self.colonies)-1
        result['recycled_slot'] = False
        return result

    def learn(self, cue):
        if self.frozen:
            raise RuntimeError('memory is frozen')
        canonical, fingerprint = self._canonical_episode(cue)
        # Journal only touched colonies, not the entire population. A failed
        # image lesson must undo earlier patch updates and funded recruitment.
        length = len(self.colonies)
        counts, support, funding = list(self.selection_counts), list(self.colony_support), dict(self.remaining_funding)
        episode_length, next_episode_id = len(self.episodes), self.next_episode_id
        counters = {name: getattr(self, name) for name in
                    ('images_seen', 'patches_seen', 'updates', 'suppressed_updates', 'novelty_overflow', 'recycled_colonies', 'motif_merges')}
        rollback = {}
        context_before = dict(self.context_associations)
        context_rows_before = dict(self._context_row_totals)
        context_targets_before = {row: dict(targets) for row, targets in self._context_row_targets.items()}
        context_counters = (self.context_updates, self.context_prediction_error)
        motif_before = dict(self.context_motifs)
        motif_rows_before = dict(self._context_motif_row_totals)
        motif_targets_before = {row: dict(targets) for row, targets in self._context_motif_row_targets.items()}
        motif_updates_before = self.context_motif_updates
        feedback_counters = (self.predictive_feedback_patches, self.predictive_feedback_error)
        self._active_recruit_rollback = rollback
        try:
            predictive_errors = self._context_prediction_errors(cue)
            result = self._learn(cue, rollback, predictive_errors)
            context = self._learn_context(cue)
            result['context_link_updates'] = context['updates']
            result['context_motif_updates'] = context['motif_updates']
            result['mean_context_surprise'] = context['mean_surprise']
            existing = self._episode_index.get(fingerprint)
            if existing is not None:
                identity = existing
                result['episode_duplicate'] = True
                result['episode_stored'] = True
            elif len(self.episodes) < self.config.max_episodes:
                identity = f"visual-{self.next_episode_id:08d}"
                compressed = zlib.compress(canonical.tobytes(order='C'), level=6)
                record = dict(identity=identity, fingerprint=fingerprint,
                              template=base64.b64encode(compressed).decode('ascii'))
                self.episodes.append(record)
                self._episode_index[fingerprint] = identity
                self._episode_template_cache = None
                self.next_episode_id += 1
                result['episode_duplicate'] = False
                result['episode_stored'] = True
            else:
                identity = None
                result['episode_duplicate'] = False
                result['episode_stored'] = False
            result['episode_identity'] = identity
            self._active_recruit_rollback = None
            return result
        except Exception:
            self._active_recruit_rollback = None
            for i, (before, exposures) in rollback.items():
                organism = self.colonies[i].organism
                env = organism.environment
                saved_env = before.pop('environment')
                env.__dict__.clear(); env.__dict__.update(saved_env.__dict__)
                organism.__dict__.clear(); organism.__dict__.update(before)
                organism.environment = env
                self.colonies[i].exposures = exposures
            del self.colonies[length:]
            del self.episodes[episode_length:]
            self._episode_index = {item['fingerprint']: item['identity'] for item in self.episodes}
            self._episode_template_cache = None
            self.context_associations = context_before
            self._context_row_totals = context_rows_before
            self._context_row_targets = context_targets_before
            self.context_updates, self.context_prediction_error = context_counters
            self.context_motifs = motif_before
            self._context_motif_row_totals = motif_rows_before
            self._context_motif_row_targets = motif_targets_before
            self.context_motif_updates = motif_updates_before
            self.predictive_feedback_patches, self.predictive_feedback_error = feedback_counters
            self.next_episode_id = next_episode_id
            for name, value in counters.items():
                setattr(self, name, value)
            self.selection_counts, self.colony_support, self.remaining_funding = counts, support, funding
            raise

    def _canonical_episode(self, cue):
        """Validate and quantize one cue for deterministic episode matching."""
        array = np.asarray(cue, dtype=float)
        expected = (self.config.size, self.config.size, 3)
        if array.shape != expected or not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
            raise ValueError('invalid visual cue')
        canonical = np.ascontiguousarray(np.rint(array * 65535).astype('<u2'))
        digest = hashlib.sha256()
        digest.update(np.asarray(canonical.shape, dtype='<u4').tobytes())
        digest.update(canonical.tobytes(order='C'))
        return canonical, digest.hexdigest()

    def _decode_episode(self, record):
        raw = zlib.decompress(base64.b64decode(record['template'], validate=True))
        expected = self.config.size * self.config.size * 3 * 2
        if len(raw) != expected:
            raise ValueError('invalid visual episode template size')
        return np.frombuffer(raw, dtype='<u2').reshape(self.config.size, self.config.size, 3)

    def recognize(self, cue):
        """Retrieve a seen visual identity or return a nearest perceptual candidate.

        Exact means the canonical visual cue matches byte-for-byte after fixed
        uint16 quantization. Perceptual candidates are ranked by RMS cue error;
        their distance is not a probability or a diagnosis.
        """
        canonical, fingerprint = self._canonical_episode(cue)
        exact = self._episode_index.get(fingerprint)
        if exact is not None:
            return dict(identity=exact, candidate_identity=exact, match_type='exact',
                        distance=0.0, margin=None, is_exact=True)
        if not self.episodes:
            return dict(identity=None, candidate_identity=None, match_type='novel',
                        distance=None, margin=None, is_exact=False)
        query = canonical.astype(np.float32) / 65535.0
        candidates = []
        for start in range(0, len(self.episodes), 256):
            records = self.episodes[start:start+256]
            bank = np.stack([self._decode_episode(record) for record in records]).astype(np.float32) / 65535.0
            distances = np.sqrt(np.mean((bank - query[None, ...]) ** 2, axis=(1, 2, 3)))
            candidates.extend((float(distance), record['identity']) for distance, record in zip(distances, records))
        candidates.sort(key=lambda item: (item[0], item[1]))
        distance, candidate_identity = candidates[0]
        margin = candidates[1][0] - distance if len(candidates) > 1 else None
        if distance <= self.config.recognition_threshold:
            return dict(identity=None, candidate_identity=candidate_identity, match_type='perceptual',
                        distance=distance, margin=margin, is_exact=False)
        return dict(identity=None, candidate_identity=candidate_identity, match_type='novel',
                    distance=distance, margin=margin, is_exact=False)

    def predict_masked_patch(self, cue, patch_index):
        """Predict one patch from the other patches using episodic context only.

        The target patch is excluded from candidate selection and cannot leak
        into the prediction. The nearest stored episode supplies the target at
        the same spatial position. This is an associative exemplar baseline,
        not an independently verified fungal prediction mechanism.
        """
        patches = self.patches(cue)
        if isinstance(patch_index, bool) or not isinstance(patch_index, int) or not 0 <= patch_index < len(patches):
            raise ValueError('patch_index is outside this image grid')
        if not self.episodes:
            raise RuntimeError('cannot predict a masked patch without stored episodes')
        visible = np.ones(len(patches), dtype=bool)
        visible[patch_index] = False
        query = np.rint(patches * 65535).astype(np.uint16)
        # Keep the image-batch axis first. Combining a slice and a boolean
        # index in one NumPy expression can move the boolean-indexed axis in
        # front of the batch axis and silently misalign candidate images.
        # Grayscale drives this score because this experiment evaluates
        # grayscale target-patch prediction; color/edge channels otherwise
        # dominate the retrieval metric without helping that target.
        query_visible = query[..., 0][visible].astype(np.float32) / 65535.0
        patch_size = self.config.patch_size
        best_distance, best_record, best_patch = math.inf, None, None
        if self._episode_template_cache is None:
            self._episode_template_cache = np.stack([self._decode_episode(record) for record in self.episodes])
        bank = self._episode_template_cache.reshape(len(self.episodes), len(patches), patch_size, patch_size, 3)
        for start in range(0, len(self.episodes), 256):
            end = min(start+256, len(self.episodes))
            bank_visible = bank[start:end][..., 0][:, visible].astype(np.float32) / 65535.0
            distances = np.sqrt(np.mean((bank_visible - query_visible[None, ...]) ** 2, axis=(1, 2, 3)))
            local = int(np.argmin(distances))
            distance = float(distances[local])
            if distance < best_distance:
                best_distance = distance
                best_index = start + local
                best_record = self.episodes[best_index]
                best_patch = bank[best_index, patch_index].astype(np.float32) / 65535.0
        return dict(predicted_patch=best_patch, candidate_identity=best_record['identity'],
                    context_rmse=best_distance)

    @staticmethod
    def _direction(source_position, target_position, side):
        sy, sx = divmod(source_position, side)
        ty, tx = divmod(target_position, side)
        delta = (ty-sy, tx-sx)
        return {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}.get(delta)

    def _rebuild_context_motif_indices(self):
        """Rebuild sparse row caches after retirement, fusion, or rollback."""
        totals, targets = {}, {}
        for (first, first_direction, second, second_direction, target), weight in self.context_motifs.items():
            row = (first, first_direction, second, second_direction)
            totals[row] = totals.get(row, 0.0) + weight
            targets.setdefault(row, {})[target] = weight
        self._context_motif_row_totals = totals
        self._context_motif_row_targets = targets

    def _learn_context(self, cue):
        """Update local neighbor predictions from the unlabeled patch field.

        Each active patch specialist predicts the specialist at an adjacent
        position. The observed co-activation reinforces the connection in
        proportion to its surprise under the source's current local transition
        distribution. No image identity or class enters this rule.
        """
        patches = self.patches(cue)
        side = int(round(math.sqrt(len(patches))))
        scores = _distances(self.colonies, patches)
        if not scores.shape[1]:
            return dict(updates=0, motif_updates=0, mean_surprise=0.0)
        assignments = np.argmin(scores, axis=1)
        assignments[~np.isfinite(np.min(scores, axis=1))] = -1
        surprises = []
        updates = 0
        for source_position in range(len(assignments)):
            source = int(assignments[source_position])
            if source < 0:
                continue
            sy, sx = divmod(source_position, side)
            for dy, dx, direction in ((-1, 0, 0), (0, 1, 1), (1, 0, 2), (0, -1, 3)):
                ty, tx = sy+dy, sx+dx
                if not (0 <= ty < side and 0 <= tx < side):
                    continue
                target_position = ty*side+tx
                target = int(assignments[target_position])
                if target < 0:
                    continue
                row_total = self._context_row_totals.get((source, direction), 0.0)
                key = (source, direction, target)
                row_targets = self._context_row_targets.setdefault((source, direction), {})
                probability = row_targets.get(target, 0.0) / row_total if row_total else 0.0
                surprise = 1.0-probability
                updated_weight = row_targets.get(target, 0.0) + surprise
                row_targets[target] = updated_weight
                self.context_associations[key] = updated_weight
                self._context_row_totals[(source, direction)] = row_total + surprise
                surprises.append(surprise)
                updates += 1
        motif_updates = 0
        if self.config.hierarchical_context:
            # Each pair of adjacent specialists jointly predicts the target
            # specialist. Direction is relative to the target, so a motif can
            # recur at different absolute image positions.
            for target_position, target_value in enumerate(assignments):
                target = int(target_value)
                if target < 0:
                    continue
                ty, tx = divmod(target_position, side)
                neighbors = []
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    sy, sx = ty+dy, tx+dx
                    if 0 <= sy < side and 0 <= sx < side:
                        source_position = sy*side+sx
                        source = int(assignments[source_position])
                        if source >= 0:
                            direction = self._direction(source_position, target_position, side)
                            neighbors.append((direction, source))
                for left, right in combinations(neighbors, 2):
                    (first_direction, first), (second_direction, second) = sorted((left, right))
                    row = (first, first_direction, second, second_direction)
                    key = row+(target,)
                    row_targets = self._context_motif_row_targets.setdefault(row, {})
                    row_total = self._context_motif_row_totals.get(row, 0.0)
                    probability = row_targets.get(target, 0.0)/row_total if row_total else 0.0
                    surprise = 1.0-probability
                    weight = row_targets.get(target, 0.0)+surprise
                    row_targets[target] = weight
                    self.context_motifs[key] = weight
                    self._context_motif_row_totals[row] = row_total+surprise
                    motif_updates += 1
        mean_surprise = float(np.mean(surprises)) if surprises else 0.0
        self.context_updates += updates
        self.context_prediction_error += float(sum(surprises))
        self.context_motif_updates += motif_updates
        return dict(updates=updates, motif_updates=motif_updates, mean_surprise=mean_surprise)

    def _context_prediction_errors(self, cue):
        """Measure pre-learning masked-patch errors without episode access."""
        patches = self.patches(cue)
        if not self.config.predictive_feedback or not self.context_associations or not self.colonies:
            return [None]*len(patches)
        scores = _distances(self.colonies, patches)
        assignments = np.full(len(patches), -1, dtype=int)
        valid = np.isfinite(np.min(scores, axis=1))
        assignments[valid] = np.argmin(scores[valid], axis=1)
        errors = []
        for patch_index, target in enumerate(patches):
            predicted = self.predict_masked_patch_from_colonies(
                cue, patch_index, patch_assignments=assignments)['predicted_patch']
            errors.append(None if predicted is None else float(np.mean((predicted-target)**2)))
        return errors

    def _consolidate_redundant_motifs(self, rollback):
        """Fuse one mutually similar specialist pair to release a funded slot.

        This is an opt-in memory-compression hypothesis, not a measured fungal
        process. It only runs at capacity. Vectorized prototype MSE keeps the
        online search tractable; current support weights transferred local traces.
        """
        threshold = self.config.motif_merge_threshold
        if threshold <= 0 or len(self.colonies) < 2:
            return None
        eligible = [i for i, colony in enumerate(self.colonies)
                    if self.colony_support[i] > 0 and colony.has_trace() and
                    any(node.alive for node in colony.organism.nodes.values())]
        if len(eligible) < 2:
            return None
        size = self.config.patch_size
        prototypes = []
        for index in eligible:
            organism = self.colonies[index].organism
            prototypes.append(np.array([
                organism.nodes[node].receptor_trace
                if organism.nodes[node].alive and organism.nodes[node].receptor_trace
                else [.5, .5, .5]
                for node in range(size*size)
            ]).reshape(size, size, 3))
        vectors = np.asarray(prototypes).reshape(len(prototypes), -1)
        squared_norm = np.einsum('ij,ij->i', vectors, vectors)
        distances = np.maximum(0.0, (squared_norm[:, None] + squared_norm[None, :] -
                                    2.0*(vectors @ vectors.T)) / vectors.shape[1])
        candidates = []
        for local_a in range(len(eligible)):
            for local_b in range(local_a+1, len(eligible)):
                a, b = eligible[local_a], eligible[local_b]
                distance = float(distances[local_a, local_b])
                if distance <= threshold:
                    candidates.append((distance, a, b))
        if not candidates:
            return None
        _, first, second = min(candidates)
        # Keep the more recently supported specialist as the recipient. Ties
        # resolve by index so checkpoints and repeated runs stay deterministic.
        recipient, donor = sorted((first, second),
                                  key=lambda i: (-self.colony_support[i], i))
        target = self.colonies[recipient]
        source = self.colonies[donor]
        if recipient not in rollback:
            rollback[recipient] = (deepcopy(target.organism.__dict__), target.exposures)
        if donor not in rollback:
            rollback[donor] = (deepcopy(source.organism.__dict__), source.exposures)
        target_org, source_org = target.organism, source.organism
        target_support = self.colony_support[recipient]
        source_support = self.colony_support[donor]
        total_support = target_support + source_support
        for node_id, target_node in target_org.nodes.items():
            source_node = source_org.nodes[node_id]
            if not target_node.alive or not source_node.alive:
                continue
            original_trace = list(target_node.receptor_trace)
            incoming_trace = list(source_node.receptor_trace)
            original_variance = list(target_node.receptor_variance)
            incoming_variance = list(source_node.receptor_variance)
            for field in ('receptor_trace', 'receptor_fast_trace'):
                left, right = getattr(target_node, field), getattr(source_node, field)
                if left and right:
                    setattr(target_node, field,
                            ((target_support*np.asarray(left) + source_support*np.asarray(right)) /
                             total_support).tolist())
                elif right:
                    setattr(target_node, field, list(right))
            if original_trace and incoming_trace and original_variance and incoming_variance:
                left, right = np.asarray(original_trace), np.asarray(incoming_trace)
                mean = (target_support*left + source_support*right) / total_support
                target_node.receptor_variance = (
                    (target_support*(np.asarray(original_variance)+(left-mean)**2) +
                     source_support*(np.asarray(incoming_variance)+(right-mean)**2)) /
                    total_support).tolist()
            target_node.memory_exposures += source_node.memory_exposures
            target_node.fast_memory_exposures += source_node.fast_memory_exposures
        source_edges = {(edge.a, edge.b): edge for edge in source_org.segments.values() if edge.alive}
        for edge in target_org.segments.values():
            other = source_edges.get((edge.a, edge.b))
            if edge.alive and other is not None:
                edge.route_trace = ((target_support*edge.route_trace + source_support*other.route_trace) /
                                    total_support)
                edge.route_fast_trace = ((target_support*edge.route_fast_trace + source_support*other.route_fast_trace) /
                                         total_support)
        target.exposures += source.exposures
        self.colony_support[recipient] = total_support
        self.colony_support[donor] = 0
        for node in source_org.nodes.values():
            node.alive = False
        for edge in source_org.segments.values():
            edge.alive = False
        for tip in source_org.tips.values():
            tip.active = False
        source.exposures = 0

        # Route evidence follows the fused identity. Combine collisions rather
        # than discarding learned transitions, then rebuild the sparse indices.
        mapped = {}
        for (source_id, direction, target_id), weight in self.context_associations.items():
            source_id = recipient if source_id == donor else source_id
            target_id = recipient if target_id == donor else target_id
            key = (source_id, direction, target_id)
            mapped[key] = mapped.get(key, 0.0) + weight
        self.context_associations = mapped
        self._context_row_totals = {}
        self._context_row_targets = {}
        for (source_id, direction, target_id), weight in mapped.items():
            row = (source_id, direction)
            self._context_row_totals[row] = self._context_row_totals.get(row, 0.0) + weight
            self._context_row_targets.setdefault(row, {})[target_id] = weight
        mapped_motifs = {}
        for (first, first_direction, second, second_direction, target_id), weight in self.context_motifs.items():
            first = recipient if first == donor else first
            second = recipient if second == donor else second
            target_id = recipient if target_id == donor else target_id
            left, right = sorted(((first_direction, first), (second_direction, second)))
            key = (left[1], left[0], right[1], right[0], target_id)
            mapped_motifs[key] = mapped_motifs.get(key, 0.0)+weight
        self.context_motifs = mapped_motifs
        self._rebuild_context_motif_indices()
        self.motif_merges += 1
        return dict(recipient=recipient, donor=donor)

    def predict_masked_patch_from_colonies(self, cue, patch_index, *, patch_assignments=None,
                                           context_order=1):
        """Predict a hidden patch from learned local colony-to-colony routes.

        This path deliberately does not consult the episodic image templates.
        It maps the visible neighboring patches to colonies, combines their
        learned directional transitions, and averages those colonies' local
        receptor traces. Returns no prediction if context has not been learned.
        """
        patches = self.patches(cue)
        if isinstance(patch_index, bool) or not isinstance(patch_index, int) or not 0 <= patch_index < len(patches):
            raise ValueError('patch_index is outside this image grid')
        if (isinstance(context_order, bool) or not isinstance(context_order, int) or
                context_order not in (1, 2)):
            raise ValueError('context_order must be 1 or 2')
        if not self.context_associations:
            return dict(predicted_patch=None, candidate_probabilities={}, context_sources=0)
        if patch_assignments is None:
            scores = _distances(self.colonies, patches)
            if not scores.shape[1]:
                return dict(predicted_patch=None, candidate_probabilities={}, context_sources=0)
            winners = np.argmin(scores, axis=1)
            winners[~np.isfinite(np.min(scores, axis=1))] = -1
        else:
            winners = np.asarray(patch_assignments)
            if (winners.shape != (len(patches),) or not np.issubdtype(winners.dtype, np.integer) or
                    np.any(winners < -1) or np.any(winners >= len(self.colonies))):
                raise ValueError('patch_assignments must contain valid colony indices or -1')
            winners = winners.astype(int, copy=True)
        # The caller may cache matches for all positions in the same image;
        # force the hidden target to be unavailable even if it was matched.
        winners[patch_index] = -1
        side = int(round(math.sqrt(len(patches))))
        py, px = divmod(patch_index, side)
        source_positions = []
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            sy, sx = py+dy, px+dx
            if 0 <= sy < side and 0 <= sx < side:
                pos = sy*side+sx
                if winners[pos] >= 0:
                    direction = self._direction(pos, patch_index, side)
                    source_positions.append((int(winners[pos]), direction))
        votes = {}
        sources = 0
        order_used = 1
        if context_order == 2 and self.config.hierarchical_context:
            motif_votes = {}
            motif_sources = 0
            for left, right in combinations(source_positions, 2):
                (first_direction, first), (second_direction, second) = sorted(
                    ((left[1], left[0]), (right[1], right[0])))
                row_key = (first, first_direction, second, second_direction)
                row = self._context_motif_row_targets.get(row_key, {})
                total = self._context_motif_row_totals.get(row_key, 0.0)
                if total <= 0:
                    continue
                motif_sources += 1
                for target, weight in row.items():
                    motif_votes[target] = motif_votes.get(target, 0.0)+weight/total
            if motif_votes:
                votes = motif_votes
                sources = motif_sources
                order_used = 2
        if not votes:
            for source, direction in source_positions:
                row = self._context_row_targets.get((source, direction), {})
                total = sum(row.values())
                if not total:
                    continue
                sources += 1
                for target, weight in row.items():
                    votes[target] = votes.get(target, 0.0) + weight/total
        if not votes:
            return dict(predicted_patch=None, candidate_probabilities={}, context_sources=0)
        total_votes = sum(votes.values())
        probabilities = {target: value/total_votes for target, value in votes.items()}
        patch_size = self.config.patch_size
        prediction = np.zeros((patch_size, patch_size, 3), dtype=float)
        for target, probability in probabilities.items():
            organism = self.colonies[target].organism
            prototype = np.array([organism.nodes[i].receptor_trace
                                  if organism.nodes[i].alive and organism.nodes[i].receptor_trace
                                  else [.5, .5, .5]
                                  for i in range(patch_size*patch_size)]).reshape(patch_size, patch_size, 3)
            prediction += probability*prototype
        return dict(predicted_patch=prediction, candidate_probabilities=probabilities,
                    context_sources=sources, context_order_used=order_used)

    def _learn(self, cue, rollback, predictive_errors=None):
        patches = self.patches(cue)
        if predictive_errors is None:
            predictive_errors = [None]*len(patches)
        if len(predictive_errors) != len(patches):
            raise ValueError('predictive feedback must match the patch grid')
        scores = _distances(self.colonies, patches)
        result = dict(patches=len(patches), updates=0, recruited=0, recycled=0, consolidated=0, suppressed=0,
                      overflow=0, synthesis_carbon=0., synthesis_energy=0.)
        feedback_errors = []
        for j, patch in enumerate(patches):
            viable = [i for i, c in enumerate(self.colonies) if any(n.alive for n in c.organism.nodes.values())]
            winner = min(viable, key=lambda i: scores[j, i]) if viable else None
            error = float(scores[j, winner]) if winner is not None else math.inf
            context_error = predictive_errors[j]
            context_surprise = (0.0 if context_error is None else
                                min(1.0, math.sqrt(max(0.0, context_error)/self.config.predictive_error_scale)))
            if context_error is not None:
                feedback_errors.append(float(context_error))
            novelty = error > self.config.novelty_threshold
            dead_slot = len(self.colonies) >= self.config.max_colonies and any(
                not any(node.alive for node in colony.organism.nodes.values()) for colony in self.colonies)
            if (novelty and not dead_slot and
                    len(self.colonies) >= self.config.max_colonies and
                    self.config.motif_merge_threshold > 0):
                merged = self._consolidate_redundant_motifs(rollback)
                if merged is not None:
                    result['consolidated'] += 1
                    # Only the fused recipient and retired donor changed; keep
                    # other prototype scores and recompute these two columns.
                    recipient, donor = merged['recipient'], merged['donor']
                    scores[j:, recipient] = _distances([self.colonies[recipient]], patches[j:])[:, 0]
                    scores[j:, donor] = np.inf
                    viable = [i for i, colony in enumerate(self.colonies)
                              if any(n.alive for n in colony.organism.nodes.values())]
                    winner = min(viable, key=lambda i: scores[j, i]) if viable else None
                    error = float(scores[j, winner]) if winner is not None else math.inf
                    novelty = error > self.config.novelty_threshold
                    dead_slot = True
            recruit = (winner is None or
                       (len(self.colonies) < self.config.max_colonies and
                        (novelty if self.config.specialization else True)) or
                       (dead_slot and novelty))
            update = self._recruit(patch) if recruit else None
            if update is not None:
                winner = update.pop('colony_index')
                recycled_slot = update.pop('recycled_slot')
                result['recycled'] += int(recycled_slot)
                result['recruited'] += 1
                replacement_scores = _distances([self.colonies[winner]], patches)[:, 0]
                if recycled_slot:
                    scores[:, winner] = replacement_scores
                else:
                    scores = np.column_stack((scores, replacement_scores))
            elif winner is not None:
                if novelty:
                    result['overflow'] += 1
                if (self.config.error_gated and error <= self.config.correction_threshold and
                        context_surprise <= self.config.correction_threshold):
                    result['suppressed'] += 1
                    self.selection_counts[winner] += 1
                    self.colony_support[winner] += 1
                    continue
                # Residual feedback avoids spending/overwriting on already learned
                # patches. Large misses receive a bounded correction, not a reset.
                reward = min(1., math.sqrt(error/self.config.novelty_threshold)) if self.config.error_gated else 1.
                reward = max(reward, context_surprise)
                if winner not in rollback:
                    colony = self.colonies[winner]
                    rollback[winner] = (deepcopy(colony.organism.__dict__), colony.exposures)
                # Already journaled for the entire image, including environment
                # identity and exposure count. Avoid nested transaction copies.
                update = self.colonies[winner]._expose(patch, reward=reward)
            else:
                raise RuntimeError('no funded colony available')
            self.selection_counts[winner] += 1
            self.colony_support[winner] += 1
            result['updates'] += 1
            result['synthesis_carbon'] += update['synthesis_carbon']
            result['synthesis_energy'] += update['synthesis_energy']
            # Later patches see corrections and new specialists from this image;
            # there is no stale fitted decoder or stale competition cache.
            if j+1 < len(patches):
                scores[j+1:, winner] = _distances([self.colonies[winner]], patches[j+1:])[:, 0]
        self.images_seen += 1
        self.patches_seen += len(patches)
        self.updates += result['updates']
        self.suppressed_updates += result['suppressed']
        self.novelty_overflow += result['overflow']
        self.recycled_colonies += result['recycled']
        result['motif_merges'] = result['consolidated']
        self.predictive_feedback_patches += len(feedback_errors)
        self.predictive_feedback_error += sum(feedback_errors)
        result['predictive_feedback_patches'] = len(feedback_errors)
        result['mean_predictive_error'] = float(np.mean(feedback_errors)) if feedback_errors else None
        return result

    def reconstruct(self, cue):
        patches = self.patches(cue)
        scores = _distances(self.colonies, patches)
        restored = np.full_like(patches, .5)
        assignment = np.full(len(patches), -1, dtype=int)
        for j in range(len(patches)):
            if not scores.shape[1] or not np.any(np.isfinite(scores[j])):
                continue
            winner = int(np.argmin(scores[j]))
            assignment[j] = winner
            o = self.colonies[winner].organism
            restored[j] = np.array([o.nodes[i].receptor_trace if o.nodes[i].alive and o.nodes[i].receptor_trace else [.5]*3
                                    for i in range(self.config.patch_size**2)]).reshape(restored[j].shape)
        p, n = self.config.patch_size, self.config.size//self.config.patch_size
        return restored.reshape(n, n, p, p, 3).transpose(0, 2, 1, 3, 4).reshape(self.config.size, self.config.size, 3), assignment

    def freeze(self):
        self.frozen = True
        for colony in self.colonies:
            colony.freeze()

    def validate(self, *, verify_episode_templates=False):
        if not isinstance(self.frozen, bool) or len(self.colonies) > self.config.max_colonies:
            raise ValueError('invalid visual population')
        for name in ('images_seen', 'patches_seen', 'updates', 'suppressed_updates', 'novelty_overflow', 'recycled_colonies', 'predictive_feedback_patches', 'motif_merges'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f'invalid {name}')
        if len(self.selection_counts) != len(self.colonies) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.selection_counts):
            raise ValueError('invalid selection counts')
        if len(self.colony_support) != len(self.colonies) or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in self.colony_support):
            raise ValueError('invalid current specialist support')
        if self.patches_seen != self.images_seen*(self.config.size//self.config.patch_size)**2 or sum(self.selection_counts) != self.patches_seen:
            raise ValueError('inconsistent image/patch counts')
        if self.updates + self.suppressed_updates != self.patches_seen:
            raise ValueError('inconsistent update counts')
        for colony in self.colonies:
            if colony.config != self._memory_config() or colony.organism.memory_frozen != self.frozen:
                raise ValueError('inconsistent colony config or freeze flag')
            if any(abs(getattr(colony.organism, 'initial_'+key)-amount) > 1e-8 for key, amount in self.unit_funding.items()):
                raise ValueError('invalid inoculum resource allocation')
            colony.organism._validate()
        for pool in (self.initial_funding, self.unit_funding, self.remaining_funding):
            if set(pool) != {'carbon', 'water', 'energy'} or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < -1e-8 for v in pool.values()):
                raise ValueError('invalid recruitment resources')
        for key in self.unit_funding:
            allocated = sum(getattr(c.organism, 'initial_'+key) for c in self.colonies)
            if abs(self.initial_funding[key] - self.unit_funding[key]*self.config.max_colonies) > 1e-8 or abs(self.initial_funding[key]-self.remaining_funding[key]-allocated) > 1e-8:
                raise ValueError('recruitment resource budget violation')
        if self.updates != sum(c.exposures for c in self.colonies):
            raise ValueError('inconsistent funded exposure count')
        if len(self.episodes) > self.config.max_episodes:
            raise ValueError('visual episode capacity exceeded')
        identities, fingerprints = set(), set()
        for index, record in enumerate(self.episodes):
            if set(record) != {'identity', 'fingerprint', 'template'}:
                raise ValueError('invalid visual episode fields')
            if (record['identity'] != f"visual-{index:08d}" or record['identity'] in identities or
                    not isinstance(record['fingerprint'], str) or len(record['fingerprint']) != 64 or
                    record['fingerprint'] in fingerprints or not isinstance(record['template'], str)):
                raise ValueError('duplicate or invalid visual episode identity')
            if verify_episode_templates:
                canonical = self._decode_episode(record)
                digest = hashlib.sha256()
                digest.update(np.asarray(canonical.shape, dtype='<u4').tobytes())
                digest.update(canonical.tobytes(order='C'))
                if digest.hexdigest() != record['fingerprint']:
                    raise ValueError('visual episode fingerprint mismatch')
            identities.add(record['identity'])
            fingerprints.add(record['fingerprint'])
        if self._episode_index != {record['fingerprint']: record['identity'] for record in self.episodes}:
            raise ValueError('visual episode index mismatch')
        if isinstance(self.next_episode_id, bool) or not isinstance(self.next_episode_id, int) or self.next_episode_id != len(self.episodes):
            raise ValueError('invalid next visual episode identity')
        if (isinstance(self.context_updates, bool) or not isinstance(self.context_updates, int) or self.context_updates < 0 or
                not isinstance(self.context_prediction_error, (int, float)) or not math.isfinite(self.context_prediction_error) or self.context_prediction_error < 0):
            raise ValueError('invalid context learning counters')
        if (not isinstance(self.predictive_feedback_error, (int, float)) or
                not math.isfinite(self.predictive_feedback_error) or self.predictive_feedback_error < 0 or
                self.predictive_feedback_patches > self.patches_seen):
            raise ValueError('invalid predictive feedback counters')
        if any(not isinstance(key, tuple) or len(key) != 3 or
               any(isinstance(v, bool) or not isinstance(v, int) for v in key) or
               not 0 <= key[0] < len(self.colonies) or key[1] not in range(4) or
               not 0 <= key[2] < len(self.colonies) or
               not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for key, value in self.context_associations.items()):
            raise ValueError('invalid local context associations')
        computed_rows = {}
        computed_targets = {}
        for (source, direction, _), value in self.context_associations.items():
            computed_rows[(source, direction)] = computed_rows.get((source, direction), 0.0)+value
            computed_targets.setdefault((source, direction), {})[_] = value
        # Relative tolerance: incremental totals and re-summed weights differ only
        # by floating-point ordering, which grows with the total's magnitude.
        if any(abs(computed_rows.get(key, 0.0)-value) > 1e-9*max(1.0, abs(value)) for key, value in self._context_row_totals.items()) or set(computed_rows) != set(self._context_row_totals):
            raise ValueError('inconsistent local context row totals')
        if computed_targets != self._context_row_targets:
            raise ValueError('inconsistent local context row index')
        if sum(self.context_associations.values()) > self.context_updates*(1+1e-9) + 1e-8:
            raise ValueError('context association count exceeds updates')
        if (isinstance(self.context_motif_updates, bool) or
                not isinstance(self.context_motif_updates, int) or self.context_motif_updates < 0):
            raise ValueError('invalid context motif learning counter')
        if any(not isinstance(key, tuple) or len(key) != 5 or
               any(isinstance(v, bool) or not isinstance(v, int) for v in key) or
               not 0 <= key[0] < len(self.colonies) or key[1] not in range(4) or
               not 0 <= key[2] < len(self.colonies) or key[3] not in range(4) or
               not 0 <= key[4] < len(self.colonies) or (key[1], key[0]) > (key[3], key[2]) or
               not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for key, value in self.context_motifs.items()):
            raise ValueError('invalid hierarchical context motifs')
        computed_motif_totals, computed_motif_targets = {}, {}
        for (first, first_direction, second, second_direction, target), value in self.context_motifs.items():
            row = (first, first_direction, second, second_direction)
            computed_motif_totals[row] = computed_motif_totals.get(row, 0.0)+value
            computed_motif_targets.setdefault(row, {})[target] = value
        if (set(computed_motif_totals) != set(self._context_motif_row_totals) or
                any(abs(computed_motif_totals[row]-self._context_motif_row_totals[row]) > 1e-10
                    for row in computed_motif_totals) or
                computed_motif_targets != self._context_motif_row_targets):
            raise ValueError('inconsistent hierarchical motif row indices')
        if sum(self.context_motifs.values()) > self.context_motif_updates + 1e-8:
            raise ValueError('context motif weight exceeds update count')
        if not self.config.hierarchical_context and self.context_motifs:
            raise ValueError('disabled hierarchy contains learned motifs')

    def summary(self):
        self.validate()
        summaries = [c.organism.summary() for c in self.colonies]
        return dict(images_seen=self.images_seen, patches_seen=self.patches_seen,
                    colonies=len(self.colonies), updates=self.updates,
                    context_connections=len(self.context_associations), context_updates=self.context_updates,
                    context_motif_connections=len(self.context_motifs),
                    context_motif_updates=self.context_motif_updates,
                    context_prediction_error=self.context_prediction_error,
                    predictive_feedback_patches=self.predictive_feedback_patches,
                    predictive_feedback_error=self.predictive_feedback_error,
                    visual_episodes=len(self.episodes), episode_capacity=self.config.max_episodes,
                    suppressed_updates=self.suppressed_updates, novelty_overflow=self.novelty_overflow,
                    recycled_colonies=self.recycled_colonies, motif_merges=self.motif_merges,
                    current_colony_support=list(self.colony_support),
                    selections=list(self.selection_counts), remaining_funding=dict(self.remaining_funding),
                    initial_funding=dict(self.initial_funding),
                    memory_material=sum(s['memory_material'] for s in summaries),
                    max_absolute_budget_error=max((abs(s[k]) for s in summaries for k in ('carbon_error', 'water_error', 'energy_error')), default=0.))

    def state_dict(self):
        associations = [[source, direction, target, weight]
                        for (source, direction, target), weight in sorted(self.context_associations.items())]
        motifs = [[first, first_direction, second, second_direction, target, weight]
                  for (first, first_direction, second, second_direction, target), weight
                  in sorted(self.context_motifs.items())]
        return dict(schema='mycelia-visual-7', config=asdict(self.config), frozen=self.frozen,
                    images_seen=self.images_seen, patches_seen=self.patches_seen, updates=self.updates,
                    suppressed_updates=self.suppressed_updates, novelty_overflow=self.novelty_overflow,
                    recycled_colonies=self.recycled_colonies,
                    motif_merges=self.motif_merges,
                    selection_counts=list(self.selection_counts), colony_support=list(self.colony_support),
                    unit_funding=dict(self.unit_funding),
                    initial_funding=dict(self.initial_funding), remaining_funding=dict(self.remaining_funding),
                    next_episode_id=self.next_episode_id,
                    episodes=[dict(record) for record in self.episodes],
                    context_associations=associations, context_updates=self.context_updates,
                    context_prediction_error=self.context_prediction_error,
                    context_motifs=motifs, context_motif_updates=self.context_motif_updates,
                    predictive_feedback_patches=self.predictive_feedback_patches,
                    predictive_feedback_error=self.predictive_feedback_error,
                    colonies=[c.state_dict() for c in self.colonies])

    def save(self, path):
        self.validate()
        path = Path(path)
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(self.state_dict(), allow_nan=False))
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        schema = data.get('schema')
        if schema not in {'mycelia-visual-1', 'mycelia-visual-2', 'mycelia-visual-3', 'mycelia-visual-4', 'mycelia-visual-5', 'mycelia-visual-6', 'mycelia-visual-7'}:
            raise ValueError('unsupported visual checkpoint')
        model = cls(VisualConfig(**data['config']))
        expected_unit = dict(model.unit_funding)
        for key in ('frozen', 'images_seen', 'patches_seen', 'updates', 'suppressed_updates', 'novelty_overflow',
                    'selection_counts', 'unit_funding', 'initial_funding', 'remaining_funding'):
            setattr(model, key, data[key])
        model.recycled_colonies = data.get('recycled_colonies', 0)
        model.motif_merges = data.get('motif_merges', 0)
        model.colonies = [MemoryColony.from_state(c) for c in data['colonies']]
        model.colony_support = list(data.get('colony_support', [0]*len(model.colonies)))
        if len(model.colony_support) != len(model.colonies):
            raise ValueError('invalid current specialist support')
        if schema in {'mycelia-visual-2', 'mycelia-visual-3', 'mycelia-visual-4', 'mycelia-visual-5', 'mycelia-visual-6', 'mycelia-visual-7'}:
            model.episodes = [dict(record) for record in data['episodes']]
            model.next_episode_id = data['next_episode_id']
            model._episode_index = {record['fingerprint']: record['identity'] for record in model.episodes}
            model._episode_template_cache = None
        if schema in {'mycelia-visual-3', 'mycelia-visual-4', 'mycelia-visual-5', 'mycelia-visual-6', 'mycelia-visual-7'}:
            model.context_associations = {(int(source), int(direction), int(target)): float(weight)
                                          for source, direction, target, weight in data['context_associations']}
            model._context_row_totals = {}
            model._context_row_targets = {}
            for (source, direction, _), weight in model.context_associations.items():
                model._context_row_totals[(source, direction)] = model._context_row_totals.get((source, direction), 0.0)+weight
                model._context_row_targets.setdefault((source, direction), {})[_] = weight
            model.context_updates = data['context_updates']
            model.context_prediction_error = data['context_prediction_error']
        if schema in {'mycelia-visual-4', 'mycelia-visual-5', 'mycelia-visual-6', 'mycelia-visual-7'}:
            model.predictive_feedback_patches = data['predictive_feedback_patches']
            model.predictive_feedback_error = data['predictive_feedback_error']
        if schema == 'mycelia-visual-7':
            model.context_motifs = {(int(first), int(first_direction), int(second),
                                     int(second_direction), int(target)): float(weight)
                                    for first, first_direction, second, second_direction, target, weight
                                    in data['context_motifs']}
            model.context_motif_updates = data['context_motif_updates']
            model._rebuild_context_motif_indices()
        if model.unit_funding != expected_unit:
            raise ValueError('invalid inoculum funding')
        model.validate(verify_episode_templates=True)
        return model
