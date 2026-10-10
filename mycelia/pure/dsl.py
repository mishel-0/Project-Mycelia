"""The operator language agents search over (sequence operators and arithmetic).

Agents do not know which operators a world uses; they enumerate these
templates (ordered by learned priors) and keep the ones the evidence supports.
"""
from __future__ import annotations

UNARY = [('repeat', k) for k in range(1, 9)] + [('reverse',), ('mirror',), ('drop_last',), ('first',)]
BINARY = [('concat_xy',), ('concat_yx',), ('around',), ('interleave',)]


def apply_unary(op, x):
    name = op[0]
    if name == 'repeat':
        return x * op[1]
    if name == 'reverse':
        return x[::-1]
    if name == 'mirror':
        return x + x[::-1]
    if name == 'drop_last':
        return x[:-1]
    if name == 'first':
        return x[:1]
    raise ValueError(op)


def apply_binary(op, x, y):
    name = op[0]
    if name == 'concat_xy':
        return x + y
    if name == 'concat_yx':
        return y + x
    if name == 'around':
        return x + y + x
    if name == 'interleave':
        out = []
        for i in range(max(len(x), len(y))):
            out += x[i:i + 1] + y[i:i + 1]
        return out
    raise ValueError(op)


def describe(op):
    return {'repeat': lambda: f'repeats the phrase {op[1]} times', 'reverse': lambda: 'reverses the phrase',
            'mirror': lambda: 'says the phrase then its reverse', 'drop_last': lambda: 'drops the last action',
            'first': lambda: 'keeps only the first action', 'concat_xy': lambda: 'does the left phrase then the right',
            'concat_yx': lambda: 'does the right phrase then the left', 'around': lambda: 'does left, right, then left again',
            'interleave': lambda: 'alternates actions from both phrases'}[op[0]]()


class Budget:
    """Counts hypothesis evaluations; every learner gets the same cap."""
    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def spend(self, n=1):
        if self.used + n > self.limit:
            raise BudgetExhausted
        self.used += n


class BudgetExhausted(Exception):
    pass


class Priors:
    """Search-order preferences learned from verified outcomes (procedural memory)."""
    def __init__(self, counts=None):
        self.counts = dict(counts or {})

    def order(self, templates):
        return sorted(templates, key=lambda t: (-self.counts.get(repr(t), 0.), templates.index(t)))

    def reinforce(self, template, amount=1.):
        self.counts[repr(template)] = self.counts.get(repr(template), 0.) + amount
