"""Receptor field embedded in a real Mycelium compartment/cord network.

Topology: a size x size lattice of actual Mycelium compartments joined by
septal cords (built like MemoryColony). Each compartment senses one image
region through the shared receptor field (unpooled responses averaged over
its region).

Unsupervised cord learning: cords whose two compartments' responses co-vary
across unlabelled images are thickened (Hebbian). Thickening is paid from the
compartments' nutrient and ATP-like energy and recorded in the organism's
ledgers; Mycelium._validate() checks the budgets after learning.

Signalling: compartment states spread along cords for a few steps, weighted
by real cord conductance (Hagen-Poiseuille, ~radius^4). The settled network
state is what TraceMemory stores (teach-only).

Biology-inspired computational hypothesis; not clinically validated.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .memory import MemoryColony, MemoryConfig
from .receptor_memory import ReceptorField


@dataclass(frozen=True)
class NetworkConfig:
    size: int = 6
    spread: float = .3
    steps: int = 2
    cord_rate: float = .2
    cord_energy_cost: float = .2
    learn_topology: bool = True


class MycelialNetwork:
    def __init__(self, field: ReceptorField, config=None):
        self.field, self.config = field, config or NetworkConfig()
        self.colony = MemoryColony(MemoryConfig(size=self.config.size))
        self.organism = self.colony.organism
        self.cord_learning = {'cords_reinforced': 0, 'carbon_invested': 0., 'energy_spent': 0.}

    def sense(self, img):
        """Per-compartment receptor responses: shape (size*size, 2*receptors)."""
        c = self.field.config; s = self.config.size
        r = self.field.sense(img)['responses']; t = c.threshold
        f = np.concatenate([np.maximum(0, r - t), np.maximum(0, -r - t)], 1)
        g = self.field._patches(img).shape[0]; f = f.reshape(g, g, -1)
        cut = np.array_split(np.arange(g), s)
        # Compartment id y*s+x matches MemoryColony's lattice layout.
        return np.stack([f[np.ix_(a, b)].mean((0, 1)) for a in cut for b in cut]).astype(np.float32)

    def learn_cords(self, sensed):
        """Unsupervised Hebbian cord thickening from co-varying compartments."""
        if not self.config.learn_topology:
            return self.cord_learning
        o = self.organism; c = self.config
        x = np.asarray(sensed, dtype=np.float64)  # images x compartments x channels
        x = x.mean(2); x = (x - x.mean(0)) / (x.std(0) + 1e-9)
        corr = (x.T @ x) / len(x)
        for e in o.segments.values():
            if not e.alive:
                continue
            a, b = o.nodes[e.a], o.nodes[e.b]
            request = c.cord_rate * max(0., float(corr[e.a, e.b]))
            invest = min(request, 2 * a.nutrient, 2 * b.nutrient,
                         2 * a.energy / c.cord_energy_cost, 2 * b.energy / c.cord_energy_cost,
                         max(0., (o.config.radius - e.radius) * 2))
            if invest <= 0:
                continue
            spent = invest * c.cord_energy_cost
            a.nutrient -= invest / 2; b.nutrient -= invest / 2; e.material += invest
            a.energy -= spent / 2; b.energy -= spent / 2; o.memory_spent += spent
            e.radius = min(o.config.radius, e.radius + invest * .5)
            self.cord_learning['cords_reinforced'] += 1
            self.cord_learning['carbon_invested'] += invest
            self.cord_learning['energy_spent'] += spent
        o._validate()
        return self.cord_learning

    def conductance_matrix(self):
        o = self.organism; n = self.config.size ** 2; w = np.zeros((n, n))
        for e in o.segments.values():
            if e.alive:
                k = e.conductance(o.config.hydraulic_scale); w[e.a, e.b] += k; w[e.b, e.a] += k
        return w / max(w.max(), 1e-30)

    def settle(self, sensed):
        """Spread compartment states along cords (conductance-weighted)."""
        c = self.config; n = len(sensed)
        out = np.empty((n, int(np.prod(sensed.shape[1:]))), dtype=np.float32)
        w = self.conductance_matrix(); w = (w / np.maximum(w.sum(1, keepdims=True), 1e-30)).astype(np.float32)
        for i in range(0, n, 256):  # chunked so memory-mapped inputs stay on disk
            x = np.asarray(sensed[i:i + 256], dtype=np.float32)
            for _ in range(c.steps if c.spread else 0):
                x = (1 - c.spread) * x + c.spread * np.einsum('ij,njc->nic', w, x)
            out[i:i + len(x)] = x.reshape(len(x), -1)
        return out

    def summary(self):
        s = self.organism.summary(); o = self.organism
        radii = [e.radius for e in o.segments.values() if e.alive]
        return {'compartments': self.config.size ** 2, 'cords': len(radii),
                'cord_radius_min': min(radii), 'cord_radius_max': max(radii),
                'carbon_error': s['carbon_error'], 'energy_error': s['energy_error'], **self.cord_learning}
