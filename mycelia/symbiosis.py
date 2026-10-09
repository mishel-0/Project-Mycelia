"""Mycelial Symbiosis Engine: plastic, budgeted, multi-timescale trace memory.

An experimental learner over receptor-network responses (vectors split into
compartment blocks). There is no fixed architecture: the set of memory traces
is the model, and it changes while learning.

Mechanisms (each with a distinct job):
  * Structural plasticity - a trace grows when recall fails or is unsure;
    traces that never help are pruned.
  * Local reward/punishment - only the traces that drove a decision get
    credit: helpful ones strengthen, misleading ones weaken (never deleted on
    one mistake, so they can recover).
  * Metabolic budget - every trace costs maintenance energy; the colony holds
    at most `budget` traces and evicts the least useful fast traces first.
  * Fast/slow memory + sleep - new traces are fast; during sleep, reliable fast
    traces merge into nearby slow traces of the same label or are promoted,
    unreliable ones are pruned.
  * Damage and self-healing - compartments can be killed (their signal is
    lost in memory and inputs); recovery replays experience through the
    surviving network to regrow traces.

Computational hypothesis inspired by fungal networks; not clinical software.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SymbiosisConfig:
    budget: int = 4000
    gamma: float = 5.
    contributors: int = 8
    reward: float = .1
    punishment: float = .2
    grow_margin: float = .2
    merge_similarity: float = .9
    sleep_every: int = 1000
    min_strength: float = .05
    max_strength: float = 4.
    plasticity: bool = True
    local_credit: bool = True
    sleep: bool = True
    seed: int = 0


class SymbiosisEngine:
    def __init__(self, dim, n_labels, compartments, config=None):
        self.c = config or SymbiosisConfig(); self.n_labels = n_labels
        self.compartments = compartments; self.block = dim // compartments
        cap = self.c.budget + 1
        self.w = np.zeros((cap, dim), dtype=np.float32); self.label = np.zeros(cap, dtype=int)
        self.strength = np.zeros(cap, dtype=np.float32); self.utility = np.zeros(cap, dtype=np.float32)
        self.slow = np.zeros(cap, dtype=bool); self.n = 0
        self.alive = np.ones(compartments, dtype=bool); self.rng = np.random.default_rng(self.c.seed)
        self.stats = dict(steps=0, grown=0, pruned=0, merged=0, promoted=0, rewards=0, punishments=0, sleeps=0,
                          energy=0.)

    # --- sensing through the (possibly damaged) network ---------------------
    def _mask(self, x):
        x = np.array(x, dtype=np.float32, ndmin=2).reshape(-1, self.compartments, self.block)
        x[:, ~self.alive] = 0
        x = x.reshape(len(x), -1)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def _activation(self, x):
        w = self.w[:self.n]
        return self.strength[:self.n] * np.exp(self.c.gamma * (x @ w.T - 1))

    def scores(self, x):
        x = self._mask(x); out = np.zeros((len(x), self.n_labels), dtype=np.float32)
        if not self.n:
            return out
        for i in range(0, len(x), 512):
            act = self._activation(x[i:i + 512])
            for k in range(self.n_labels):
                out[i:i + 512, k] = act[:, self.label[:self.n] == k].sum(1)
        return out

    def predict(self, x):
        return self.scores(x).argmax(1)

    # --- online learning ------------------------------------------------------
    def _grow(self, x, y):
        if self.n >= self.c.budget:
            self._evict()
        i = self.n; self.w[i] = x; self.label[i] = y; self.strength[i] = 1.; self.utility[i] = 0.; self.slow[i] = False
        self.n += 1; self.stats['grown'] += 1

    def _remove(self, keep):
        k = int(keep.sum()); self.stats['pruned'] += self.n - k
        for a in (self.w, self.label, self.strength, self.utility, self.slow):
            a[:k] = a[:self.n][keep]
        self.n = k

    def _evict(self):
        # Least useful trace goes first; slow traces are protected unless no fast ones remain.
        value = self.utility[:self.n] + self.strength[:self.n] + 1e3 * self.slow[:self.n]
        keep = np.ones(self.n, dtype=bool); keep[int(np.argmin(value))] = False
        self._remove(keep)

    def learn(self, x, y):
        x = self._mask(x)[0]; c = self.c; self.stats['steps'] += 1
        self.stats['energy'] += self.n  # maintenance + recall cost, one unit per trace consulted
        if not self.n:
            self._grow(x, y); return 'grow'
        act = self._activation(x[None])[0]
        totals = np.bincount(self.label[:self.n], weights=act, minlength=self.n_labels)
        guess = int(totals.argmax()); share = totals / max(totals.sum(), 1e-12)
        margin = share[y] - np.max(np.delete(share, y))
        if c.local_credit:
            top = np.argsort(-act)[:c.contributors]; part = act[top] / max(act[top].sum(), 1e-12)
            good = self.label[top] == y
            self.strength[top[good]] += c.reward * part[good]; self.utility[top[good]] += part[good]
            self.strength[top[~good]] -= c.punishment * part[~good]; self.utility[top[~good]] -= part[~good]
            np.clip(self.strength[:self.n], c.min_strength, c.max_strength, out=self.strength[:self.n])
            self.stats['rewards'] += int(good.sum()); self.stats['punishments'] += int((~good).sum())
        outcome = 'correct' if guess == y else 'error'
        if c.plasticity and (guess != y or margin < c.grow_margin):
            self._grow(x, y); outcome += '+grow'
        elif not c.plasticity and self.n < c.budget:
            self._grow(x, y)
        if c.sleep and self.stats['steps'] % c.sleep_every == 0:
            self.sleep()
        return outcome

    def fit_stream(self, xs, ys, epochs=1):
        order = np.arange(len(xs))
        for _ in range(epochs):
            self.rng.shuffle(order)
            for i in order:
                self.learn(xs[i], ys[i])
        return self

    # --- sleep: consolidation and pruning ---------------------------------
    def sleep(self):
        c = self.c; self.stats['sleeps'] += 1
        if not self.n:
            return
        keep = np.ones(self.n, dtype=bool)
        fast = np.where(~self.slow[:self.n])[0]; slow = np.where(self.slow[:self.n])[0]
        for i in fast:
            if self.utility[i] < 0 and self.strength[i] <= c.min_strength * 2:
                keep[i] = False; continue  # misleading and weak: pruned
            if self.utility[i] <= 0:
                continue  # unproven: stays fast for now
            same = slow[self.label[slow] == self.label[i]]
            if len(same):
                sim = self.w[same] @ self.w[i]; j = same[int(sim.argmax())]
                if sim.max() >= c.merge_similarity:
                    a, b = self.strength[j], self.strength[i]
                    v = a * self.w[j] + b * self.w[i]; self.w[j] = v / max(np.linalg.norm(v), 1e-12)
                    self.strength[j] = min(c.max_strength, a + .5 * b); self.utility[j] += self.utility[i]
                    keep[i] = False; self.stats['merged'] += 1; continue
            self.slow[i] = True; self.stats['promoted'] += 1; slow = np.append(slow, i)
        self._remove(keep)
        self.utility[:self.n] *= .5  # utility decays; old merit must be re-earned

    # --- damage -----------------------------------------------------------
    def damage_compartments(self, fraction, rng=None):
        rng = rng or self.rng; alive = np.where(self.alive)[0]
        dead = rng.choice(alive, int(round(fraction * self.compartments)), replace=False)
        self.alive[dead] = False
        return dead

    def damage_traces(self, fraction, rng=None):
        rng = rng or self.rng; keep = np.ones(self.n, dtype=bool)
        keep[rng.choice(self.n, int(round(fraction * self.n)), replace=False)] = False
        pruned = self.stats['pruned']; self._remove(keep); self.stats['pruned'] = pruned

    def heal(self, xs, ys, epochs=1):
        """Self-healing: stored traces are re-expressed through surviving
        compartments, then experience is replayed so failures regrow traces."""
        w = self.w[:self.n].reshape(self.n, self.compartments, self.block); w[:, ~self.alive] = 0
        norms = np.linalg.norm(self.w[:self.n], axis=1, keepdims=True); self.w[:self.n] /= np.maximum(norms, 1e-12)
        return self.fit_stream(xs, ys, epochs)

    def summary(self):
        return dict(traces=int(self.n), slow=int(self.slow[:self.n].sum()), fast=int((~self.slow[:self.n]).sum()),
                    alive_compartments=int(self.alive.sum()), **self.stats)
