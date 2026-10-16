"""Rule world: a hidden arithmetic rule f(a, b) seen through a few examples."""
from __future__ import annotations
import itertools, random

LEAVES = ['a', 'b', 1, 2, 3]
OPS = ['+', '-', '*', 'max', 'min']
# Hidden rules are drawn from a skewed family so learned priors can matter.
FAMILY = [('+', 'a', 'b'), ('*', 'a', 'b'), ('-', 'a', 'b'), ('+', 'a', ('*', 2, 'b')), ('max', 'a', 'b'),
          ('*', ('+', 'a', 'b'), 2), ('+', ('*', 'a', 'b'), 1), ('min', 'a', 'b'), ('-', ('*', 'a', 3), 'b')]
FAMILY_WEIGHTS = [6, 5, 3, 3, 2, 2, 2, 1, 1]


def evaluate(expr, a, b):
    if expr == 'a':
        return a
    if expr == 'b':
        return b
    if isinstance(expr, int):
        return expr
    op, x, y = expr; x, y = evaluate(x, a, b), evaluate(y, a, b)
    return {'+': x + y, '-': x - y, '*': x * y, 'max': max(x, y), 'min': min(x, y)}[op]


def expressions(depth=2):
    """All expressions up to depth, shortest first (the inducer's search space)."""
    levels = [list(LEAVES)]
    for _ in range(depth):
        prev = [e for lvl in levels for e in lvl]
        levels.append([(op, x, y) for op in OPS for x in prev for y in prev if not (isinstance(x, int) and isinstance(y, int))])
    return [e for lvl in levels for e in lvl]


def shape(expr):
    """Vocabulary-free template of an expression (for procedural memory)."""
    return repr(expr)


class RuleEpisode:
    def __init__(self, seed, n_obs=5, noise=False):
        rng = random.Random(seed); self.rule = rng.choices(FAMILY, FAMILY_WEIGHTS)[0]
        pts = rng.sample(list(itertools.product(range(0, 10), range(0, 10))), n_obs + 20)
        self.obs = [(a, b, evaluate(self.rule, a, b)) for a, b in pts[:n_obs]]
        self.noisy_index = None
        if noise:
            a, b, y = self.obs[0]; self.obs[0] = (a, b, y + rng.choice([-3, -2, 2, 3])); self.noisy_index = 0
        self.tests = [(a, b, evaluate(self.rule, a, b)) for a, b in pts[n_obs:]]
