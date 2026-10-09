"""Receptor-field memory: unsupervised hyphal receptors + teach-only recall.

Stage 1 (unsupervised, no labels): hyphal "tips" compete for local 6x6 image
patches. Each tip moves toward the patches it wins (competitive / spherical
k-means learning), so the colony grows a dictionary of local receptors.
Responses are pooled over 1x1, 2x2 and GxG regions (cord-like aggregation).

Stage 2 (teach-only): every labelled training image, and its mirror image, is
stored as a memory trace. Recall weights stored traces by an exponential
similarity kernel with weights from one closed-form linear solve (kernel
ridge). No CNN, no backpropagation, no reward/punishment.

This is a computational, biology-inspired hypothesis, not evidence of fungal
image cognition, and not a clinically validated diagnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReceptorConfig:
    receptors: int = 800
    patch: int = 6
    stride: int = 1
    grid: int = 4
    patches_per_image: int = 20
    iterations: int = 15
    whitening: float = .1
    threshold: float = .25
    seed: int = 0


def _norm(p):
    p = p - p.mean(-1, keepdims=True)
    return p / np.sqrt(p.var(-1, keepdims=True) + .01)


class ReceptorField:
    """Label-free receptor dictionary learned by competing hyphal tips."""

    def __init__(self, config=None):
        self.config = config or ReceptorConfig()
        self.mu = self.whiten = self.receptors = None

    def _patches(self, img):
        c = self.config
        w = np.lib.stride_tricks.sliding_window_view(img, (c.patch, c.patch))[::c.stride, ::c.stride]
        return w.reshape(w.shape[0], w.shape[1], -1)

    def fit(self, images):
        """Unsupervised: images only, labels are never seen."""
        c = self.config; rng = np.random.default_rng(c.seed)
        samples = []
        for img in images:
            p = self._patches(img).reshape(-1, c.patch * c.patch)
            samples.append(p[rng.choice(len(p), c.patches_per_image, replace=False)])
        s = _norm(np.concatenate(samples))
        self.mu = s.mean(0)
        e, v = np.linalg.eigh(np.cov(s - self.mu, rowvar=False))
        self.whiten = v @ np.diag(1 / np.sqrt(e + c.whitening)) @ v.T
        z = (s - self.mu) @ self.whiten
        d = z[rng.choice(len(z), c.receptors, replace=False)]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        for _ in range(c.iterations):
            # Tips compete for patches; each winner grows toward what it captured.
            winner = np.argmax(z @ d.T, axis=1)
            for k in range(c.receptors):
                won = z[winner == k]
                d[k] = won.sum(0) if len(won) else z[rng.integers(len(z))]
            d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        self.receptors = d
        return self

    def encode(self, img):
        c = self.config; p = self._patches(img); g = p.shape[0]
        r = ((_norm(p.reshape(-1, c.patch * c.patch)) - self.mu) @ self.whiten) @ self.receptors.T
        f = np.concatenate([np.maximum(0, r - c.threshold), np.maximum(0, -r - c.threshold)], 1).reshape(g, g, -1)
        out = []
        for grid in sorted({1, 2, c.grid}):
            cut = np.array_split(np.arange(g), grid)
            out += [f[np.ix_(a, b)].mean((0, 1)) for a in cut for b in cut]
        return np.concatenate(out).astype(np.float32)

    def sense(self, img):
        """Unpooled receptor responses plus each window's brightness and contrast."""
        c = self.config; p = self._patches(img); g = p.shape
        flat = p.reshape(-1, c.patch * c.patch)
        mean = flat.mean(1, keepdims=True); sd = np.sqrt(flat.var(1, keepdims=True) + .01)
        responses = (((flat - mean) / sd - self.mu) @ self.whiten) @ self.receptors.T
        return {'responses': responses, 'mean': mean, 'sd': sd, 'grid': g[:2], 'shape': np.shape(img)}

    def reconstruct(self, sensed, keep=None):
        """Rebuild the image from sensed responses.

        keep=None uses every receptor; the receptors span all patch shapes, so
        this is exact (lossless after uint8 rounding). keep=k uses only the k
        strongest receptors per window (lossy, k >= patch*patch is exact).
        """
        c = self.config; r = sensed['responses']; d = self.receptors
        if keep is None:
            z = r @ np.linalg.pinv(d.T)
        else:
            idx = np.argsort(-np.abs(r), 1)[:, :keep]
            z = np.empty((len(r), d.shape[1]))
            for i, sel in enumerate(idx):
                coef, *_ = np.linalg.lstsq(d[sel] @ d[sel].T, r[i, sel], rcond=None)
                z[i] = coef @ d[sel]
        patches = ((z @ np.linalg.inv(self.whiten)) + self.mu) * sensed['sd'] + sensed['mean']
        out = np.zeros(sensed['shape']); count = np.zeros(sensed['shape'])
        gy, gx = sensed['grid']; patches = patches.reshape(gy, gx, c.patch, c.patch)
        for dy in range(c.patch):
            for dx in range(c.patch):
                ys = slice(dy, dy + (gy - 1) * c.stride + 1, c.stride); xs = slice(dx, dx + (gx - 1) * c.stride + 1, c.stride)
                out[ys, xs] += patches[:, :, dy, dx]; count[ys, xs] += 1
        return np.clip(out / np.maximum(count, 1), 0, 1)

    def encode_many(self, images):
        return np.stack([self.encode(i) for i in images])


class TraceMemory:
    """Teach-only memory over receptor responses (exponential-kernel recall)."""

    def __init__(self, n_labels, gamma=2., lam=.003):
        self.n_labels, self.gamma, self.lam = n_labels, gamma, lam

    def _n(self, x):
        x = (x - self.mean) / self.scale
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def fit(self, features, labels, extra=()):
        """features: training responses; extra: (features, labels) mirror imprints."""
        self.mean = features.mean(0, dtype=np.float64).astype(np.float32)
        self.scale = (features.std(0, dtype=np.float64) + 1e-6).astype(np.float32)
        parts = [self._n(features)] + [self._n(f) for f, _ in extra]
        self.traces = np.concatenate(parts)
        y = np.concatenate([labels] + [l for _, l in extra])
        k = self.traces @ self.traces.T; k -= 1; k *= self.gamma; np.exp(k, out=k)
        k[np.diag_indices_from(k)] += self.lam
        self.weights = np.linalg.solve(k, (np.eye(self.n_labels)[y] * 2 - 1).astype(np.float32))
        return self

    def scores(self, features):
        k = self._n(features) @ self.traces.T; k -= 1; k *= self.gamma; np.exp(k, out=k)
        return k @ self.weights

    def predict(self, features):
        return self.scores(features).argmax(1)
