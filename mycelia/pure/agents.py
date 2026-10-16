"""Rule-based agents. Each sees the demonstrations plus whatever verified
knowledge the coordinator shares with it, and spends evaluations from a budget."""
from __future__ import annotations
from itertools import permutations
from .bus import Message
from .dsl import UNARY, BINARY, apply_unary, apply_binary, BudgetExhausted


def execute(K, phrase):
    """Executor: run a phrase with learned knowledge K; None if any word is unknown."""
    def term(ws):
        if ws[0] not in K['prims']:
            return None
        x = list(K['prims'][ws[0]])
        if len(ws) == 1:
            return x
        return apply_unary(K['unary'][ws[1]], x) if len(ws) == 2 and ws[1] in K['unary'] else None
    for i, w in enumerate(phrase):
        if w in K['binary']:
            x, y = term(phrase[:i]), term(phrase[i + 1:])
            return None if x is None or y is None else apply_binary(K['binary'][w], x, y)
    return term(phrase) if len(phrase) <= 2 else None


class Observer:
    name = 'observer'

    def run(self, demos, bus, memory, episode):
        for i, (p, out) in enumerate(demos):
            memory.add('episodic', ' '.join(p), out, 'observed', self.name, episode, [i])
            bus.post(Message(self.name, 'observation', {'phrase': p, 'output': out}, [i], 1., 'observed'))


class Lexicon:
    """Language agent: primitives from single-word demos; inverts known operators to
    recover primitives that were never shown alone."""
    name = 'lexicon'

    def propose(self, K, demos, budget, use_operators=True):
        out, known = [], K['prims']
        for i, (p, o) in enumerate(demos):
            if len(p) == 1 and p[0] not in known:
                budget.spend(); out.append(('prims', p[0], tuple(o), [i]))
        if not use_operators:
            return out
        tokens = sorted({t for _, o in demos for t in o})
        for i, (p, o) in enumerate(demos):
            if len(p) == 2 and p[0] not in known and p[1] in K['unary']:
                for cand in permutations(tokens, 2):
                    budget.spend()
                    if apply_unary(K['unary'][p[1]], list(cand)) == o:
                        out.append(('prims', p[0], cand, [i])); break
        return out


class RuleInducer:
    """Reasoning agent: searches the operator language for each function word,
    in the order given by learned priors."""
    name = 'inducer'

    def __init__(self, priors):
        self.priors = priors

    def propose(self, K, demos, budget):
        out = []
        for slot, size, templates, apply in (('unary', 2, UNARY, None), ('binary', 3, BINARY, None)):
            words = {p[1] for p, _ in demos if len(p) == size and p[1] not in K[slot] and p[1] not in K['prims']}
            for w in sorted(words):
                usable = [(i, p, o) for i, (p, o) in enumerate(demos) if len(p) == size and p[1] == w
                          and all(x in K['prims'] for x in (p[0],) + ((p[2],) if size == 3 else ()))]
                if not usable:
                    continue
                best = None
                for op in self.priors.order(list(templates)):
                    ok = []
                    for i, p, o in usable:
                        budget.spend()
                        x = list(K['prims'][p[0]])
                        got = apply_unary(op, x) if size == 2 else apply_binary(op, x, list(K['prims'][p[2]]))
                        ok.append(got == o)
                    if all(ok):
                        best = (slot, w, op, [i for i, _, _ in usable], []); break
                    if best is None or sum(ok) > len(best[3]):
                        best = (slot, w, op, [u[0] for u, g in zip(usable, ok) if g], [u[0] for u, g in zip(usable, ok) if not g])
                if best:
                    out.append(best if len(best) == 5 else best + ([],))
        return out


class Critic:
    """Checks hypotheses against every demo it can evaluate; tolerates a single
    outlier only when the rest strongly agree, and flags that demo as suspect."""
    name = 'critic'

    def __init__(self, tolerant=True, min_support=2):
        self.tolerant, self.min_support = tolerant, min_support

    def review(self, slot, word, meaning, K, demos, budget):
        trial = {k: dict(v) for k, v in K.items()}; trial[slot][word] = meaning
        agree, disagree = [], []
        for i, (p, o) in enumerate(demos):
            if word in p:
                budget.spend(); got = execute(trial, p)
                if got is not None:
                    (agree if got == list(o) else disagree).append(i)
        if not disagree and agree:
            return 'verified', agree, []
        if self.tolerant and len(disagree) == 1 and len(agree) >= self.min_support:
            return 'verified', agree, disagree
        return 'refuted', agree, disagree
