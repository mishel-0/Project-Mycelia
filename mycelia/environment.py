from __future__ import annotations

import math
import numpy as np


class Environment:
    """Finite, nonnegative nutrient and water amounts in square substrate cells.

    Coordinates are grid units. Edge boundaries are reflecting. Temperature is
    Celsius; the remaining quantities use the documented model units.
    """

    def __init__(self, nutrient, water=8.0, *, diffusion=0.04, temperature=25.0):
        self.nutrient = np.array(nutrient, dtype=float, copy=True)
        if self.nutrient.ndim != 2 or min(self.nutrient.shape) < 3:
            raise ValueError("nutrient must be a 2D array with each dimension >= 3")
        if not np.all(np.isfinite(self.nutrient)) or np.any(self.nutrient < 0):
            raise ValueError("nutrient amounts must be finite and nonnegative")
        self.water = np.broadcast_to(np.asarray(water, dtype=float), self.shape).copy()
        if not np.all(np.isfinite(self.water)) or np.any(self.water < 0):
            raise ValueError("water amounts must be finite and nonnegative")
        if not math.isfinite(diffusion) or diffusion < 0 or not math.isfinite(temperature):
            raise ValueError("invalid environmental parameters")
        self.diffusion = float(diffusion)
        self.temperature = float(temperature)
        self.oxygen = np.ones(self.shape)
        self.stress = np.zeros(self.shape)
        self.resistance = np.zeros(self.shape)
        self.blocked = np.zeros(self.shape, dtype=bool)

    @property
    def shape(self):
        return self.nutrient.shape

    def cell(self, x, y):
        h, w = self.shape
        return int(np.clip(round(y), 0, h-1)), int(np.clip(round(x), 0, w-1))

    def inside(self, x, y):
        h, w = self.shape
        return 0 <= x <= w-1 and 0 <= y <= h-1

    def sample(self, x, y):
        return float(self.nutrient[self.cell(x, y)])

    def concentration(self, x, y):
        cell = self.cell(x, y)
        return float(self.nutrient[cell] / max(self.water[cell], 1e-12))

    def gradient(self, x, y):
        # Zero remains a zero cue. No synthetic default direction.
        return ((self.sample(x+1, y)-self.sample(x-1, y))/2,
                (self.sample(x, y+1)-self.sample(x, y-1))/2)

    def path_clear(self,x0,y0,x1,y1):
        count=max(1,math.ceil(math.hypot(x1-x0,y1-y0)/.2))
        for fraction in np.linspace(0,1,count+1):
            x=x0+(x1-x0)*float(fraction)
            y=y0+(y1-y0)*float(fraction)
            if not self.inside(x,y) or self.blocked[self.cell(x,y)]:
                return False
        return True

    def consume(self, x, y, amount, *, pool="nutrient"):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("amount must be finite and nonnegative")
        field = getattr(self, pool)
        cell = self.cell(x, y)
        actual = min(float(field[cell]), float(amount))
        field[cell] -= actual
        return actual

    def deposit(self, x, y, amount, *, pool="nutrient"):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("amount must be finite and nonnegative")
        getattr(self, pool)[self.cell(x, y)] += amount

    def step(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Conservative pairwise face exchange, with substeps for positivity.
        count = max(1, math.ceil(self.diffusion*dt/0.24))
        rate = self.diffusion*dt/count
        for _ in range(count):
            a = self.nutrient
            delta = np.zeros_like(a)
            horizontal = rate*(a[:, :-1]-a[:, 1:])
            vertical = rate*(a[:-1, :]-a[1:, :])
            delta[:, :-1] -= horizontal
            delta[:, 1:] += horizontal
            delta[:-1, :] -= vertical
            delta[1:, :] += vertical
            self.nutrient += delta

    def add_patch(self, x, y, radius=6.0, strength=3.0):
        if radius <= 0 or strength < 0 or not all(math.isfinite(v) for v in (x,y,radius,strength)):
            raise ValueError("invalid patch")
        yy, xx = np.indices(self.shape)
        added = strength*np.exp(-((xx-x)**2+(yy-y)**2)/(2*radius**2))
        self.nutrient += added
        return float(added.sum())


def scenario(name="patches", size=64, diffusion=0.04):
    if size < 12:
        raise ValueError("scenario size must be >= 12")
    env = Environment(np.full((size,size), 0.02 if name != "uniform" else 2.0), diffusion=diffusion)
    if name == "patches":
        for x,y,s in ((.38,.35,3.0),(.7,.65,4.0),(.24,.75,2.5)):
            env.add_patch(x*size,y*size,radius=size*.13,strength=s)
    elif name == "obstacle":
        env.add_patch(size*.7,size*.5,size*.13,5.0)
        wall_x = int(size*.58)
        env.blocked[size//3:2*size//3, wall_x:wall_x+2] = True
    elif name not in ("uniform", "poor"):
        raise ValueError(f"unknown scenario: {name}")
    return env
