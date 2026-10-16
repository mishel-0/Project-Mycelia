"""Rule-world inducer + critic: find f(a, b) from a few examples."""
from __future__ import annotations
from .dsl import Budget, BudgetExhausted
from .worlds.rules import expressions, evaluate, shape

SPACE = expressions(2)


class RuleLearner:
    def __init__(self, priors=None, critic=True):
        self.priors, self.critic = priors, critic

    def order(self):
        if not self.priors:
            return SPACE
        weight = self.priors.counts
        return sorted(SPACE, key=lambda e: -weight.get(shape(e), 0.))

    def learn(self, ep, budget_limit, learn_priors=False):
        budget = Budget(budget_limit); found, outlier = None, None
        try:
            for e in self.order():
                budget.spend()
                bad = [i for i, (a, b, y) in enumerate(ep.obs) if evaluate(e, a, b) != y]
                if not bad:
                    found = e; break
                if self.critic and len(bad) == 1 and len(ep.obs) - 1 >= 3 and found is None:
                    found, outlier = e, bad[0]; break
        except BudgetExhausted:
            pass
        if found is not None and learn_priors and self.priors is not None:
            self.priors.counts[shape(found)] = self.priors.counts.get(shape(found), 0) + 1
        return {'rule': found, 'outlier': outlier, 'evaluations': budget.used,
                'accuracy': None if found is None else sum(evaluate(found, a, b) == y for a, b, y in ep.tests) / len(ep.tests)}
