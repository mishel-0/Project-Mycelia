import numpy as np
from mycelia.symbiosis import SymbiosisEngine, SymbiosisConfig


def data(seed, n=200):
    rng = np.random.default_rng(seed); centers = rng.standard_normal((4, 6, 5))
    y = rng.integers(0, 4, n); x = centers[y] + .4 * rng.standard_normal((n, 6, 5))
    return x.reshape(n, -1).astype(np.float32), y


def test_engine_learns_compactly():
    x, y = data(0); xt, yt = data(0, 300)
    e = SymbiosisEngine(30, 4, 6, SymbiosisConfig(gamma=10., sleep_every=50)).fit_stream(x, y)
    assert (e.predict(xt) == yt).mean() > .9 and e.n < len(x)
    s = e.summary(); assert s['sleeps'] > 0 and s['slow'] > 0


def test_budget_is_never_exceeded():
    x, y = data(1)
    e = SymbiosisEngine(30, 4, 6, SymbiosisConfig(budget=10, grow_margin=1.)).fit_stream(x, y)
    assert e.n <= 10


def test_damage_and_heal():
    x, y = data(2); e = SymbiosisEngine(30, 4, 6, SymbiosisConfig(gamma=10.)).fit_stream(x, y)
    dead = e.damage_compartments(.5, np.random.default_rng(0))
    assert len(dead) == 3 and e.alive.sum() == 3
    e.heal(x, y)
    w = e.w[:e.n].reshape(e.n, 6, 5)
    assert np.all(w[:, dead] == 0) and (e.predict(x) == y).mean() > .8
    n = e.n; e.damage_traces(.5, np.random.default_rng(0)); assert e.n == n - round(.5 * n)
