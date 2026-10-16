"""Product-quantized codebook: a node's state becomes L token ids (each < K)."""
from __future__ import annotations
import hashlib
import numpy as np


def kmeans(x, k, seed=0, iters=50):
    rng = np.random.default_rng(seed); k = min(k, len(np.unique(x, axis=0)))
    c = x[rng.choice(len(x), 1)]
    while len(c) < k:  # k-means++ seeding
        d = ((x[:, None] - c[None]) ** 2).sum(-1).min(1)
        c = np.vstack([c, x[rng.choice(len(x), p=d / d.sum())]]) if d.sum() > 0 else np.vstack([c, x[rng.integers(len(x))]])
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[a == j].mean(0) if np.any(a == j) else c[j] for j in range(len(c))])
        if np.allclose(new, c):
            break
        c = new
    return c


class Codebook:
    def __init__(self, k, parts, seed=0):
        if not 1 <= k <= 256:
            raise ValueError('K must fit one byte per token')
        self.k, self.parts, self.seed = k, parts, seed

    def fit(self, h):
        h = np.asarray(h, dtype=np.float64); self.mean = h.mean(0); self.scale = h.std(0) + 1e-9
        z = (h - self.mean) / self.scale; self.splits = np.array_split(np.arange(h.shape[1]), self.parts)
        self.centers = [kmeans(z[:, s], self.k, self.seed + i) for i, s in enumerate(self.splits)]
        blob = np.concatenate([c.ravel() for c in self.centers] + [self.mean, self.scale]).tobytes()
        self.id = int(hashlib.sha256(blob).hexdigest()[:2], 16)  # 1-byte vocabulary id carried in the envelope
        return self

    def encode(self, h):
        z = (np.asarray(h, dtype=np.float64) - self.mean) / self.scale
        return np.stack([((z[:, s, None] - c.T[None]) ** 2).sum(1).argmin(1) for s, c in zip(self.splits, self.centers)], 1)

    def decode(self, tokens):
        """Token ids -> state vector in original units (what a receiver sees)."""
        z = np.zeros((len(tokens), len(self.mean)))
        for j, (s, c) in enumerate(zip(self.splits, self.centers)):
            z[:, s] = c[np.clip(tokens[:, j], 0, len(c) - 1)]
        return z * self.scale + self.mean

    @property
    def bits(self):
        return self.parts * int(np.ceil(np.log2(max(self.k, 2))))
