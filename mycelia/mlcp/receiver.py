"""Coordinator node: fuses decoded messages, diagnoses, and decides to abstain.

Missing (or corrupt) messages arrive as None; their slot is zeroed and a
missing flag is set, so the receiver learns to cope with absent senders.
"""
from __future__ import annotations
import numpy as np


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


class Receiver:
    def __init__(self, n_labels=4, l2=1e-2, iters=2000, lr=.5, coverage=.9, seed=0):
        self.n, self.l2, self.iters, self.lr, self.coverage, self.seed = n_labels, l2, iters, lr, coverage, seed

    def _features(self, states, missing):
        x = np.column_stack([np.where(m[:, None], 0., (s - mu) / sd) for s, m, mu, sd in zip(states, missing, self.mu, self.sd)])
        return np.column_stack([x, np.column_stack(missing).astype(float)])

    def _fit_softmax(self, x, y):
        w = np.zeros((x.shape[1], self.n)); b = np.zeros(self.n); t = np.eye(self.n)[y]
        for _ in range(self.iters):
            g = _softmax(x @ w + b) - t; w -= self.lr * (x.T @ g / len(x) + self.l2 * w); b -= self.lr * g.mean(0)
        return w, b

    def _risk_features(self, p, missing, leans):
        top = np.sort(p, 1); dis = np.array([len(set(r[~m])) if (~m).any() else 0 for r, m in zip(leans, missing)])
        return np.column_stack([top[:, -1], top[:, -1] - top[:, -2], missing.sum(1), dis])

    def fit(self, states, missing, y, leans, drop_rate=.1):
        rng = np.random.default_rng(self.seed); missing = [m.copy() for m in missing]
        self.mu = [s[~m].mean(0) if (~m).any() else np.zeros(s.shape[1]) for s, m in zip(states, missing)]
        self.sd = [s[~m].std(0) + 1e-9 if (~m).any() else np.ones(s.shape[1]) for s, m in zip(states, missing)]
        for m in missing:  # train with random sender loss so missing messages are handled
            m |= rng.random(len(m)) < drop_rate
        x = self._features(states, missing); M = np.column_stack(missing)
        self.w, self.b = self._fit_softmax(x, y)
        oof = np.zeros((len(y), self.n)); folds = rng.permutation(len(y)) % 2
        for f in (0, 1):  # cross-fitted errors: risk is learned from honest mistakes
            w, b = self._fit_softmax(x[folds != f], y[folds != f]); oof[folds == f] = _softmax(x[folds == f] @ w + b)
        r = self._risk_features(oof, M, leans); wrong = (oof.argmax(1) != y).astype(float)
        self.rm, self.rs = r.mean(0), r.std(0) + 1e-9; z = (r - self.rm) / self.rs
        self.rw, self.rb = np.zeros(z.shape[1]), 0.
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(z @ self.rw + self.rb))); g = p - wrong
            self.rw -= self.lr * (z.T @ g / len(g) + self.l2 * self.rw); self.rb -= self.lr * g.mean()
        self.threshold = float(np.quantile(self.risk(oof, M, leans), self.coverage))
        return self

    def risk(self, p, missing, leans):
        z = (self._risk_features(p, missing, leans) - self.rm) / self.rs
        return 1 / (1 + np.exp(-(z @ self.rw + self.rb)))

    def decide(self, states, missing, leans):
        p = _softmax(self._features(states, missing) @ self.w + self.b); M = np.column_stack(missing)
        risk = self.risk(p, M, leans)
        return {'probabilities': p, 'prediction': p.argmax(1), 'risk': risk, 'abstain': risk > self.threshold}
