"""Solve new string tasks by composing learned skills (bounded program search).

Search order = relevance of the goal's words to each skill's documentation
words, plus procedural priors from earlier verified solutions. A program must
reproduce every visible example; if none is found within budget, the answer
is 'not enough evidence' rather than a guess.
"""
from __future__ import annotations
import heapq, itertools, re
from .dsl import Budget, BudgetExhausted
from .sandbox import run

WORD = re.compile(r'[a-z]+')


def stem(w):
    return w[:4]  # shared 4-letter beginning: swap~swapcase, capital~capitalize, lower~lowercase


class Solver:
    def __init__(self, skills, constants=(), use_docs=True, priors=None):
        self.skills = [s for s in skills if s.get('status', 'verified') in ('verified', 'observed', 'unverified', 'contested')]
        self.constants, self.use_docs = list(constants), use_docs
        self.priors = priors if priors is not None else {}

    def relevance(self, skill, goal_words):
        score = 0.
        if self.use_docs:
            words = {stem(w) for w in skill.get('words', [])}; name = stem(skill['method'])
            score += sum(2 * (stem(g) == name) + (stem(g) in words) for g in goal_words)  # a name match counts double
        for g in goal_words:
            score += 2 * self.priors.get((stem(g), skill['method']), 0)
        return score

    def steps(self, task):
        goal = [w for w in WORD.findall(task['goal'].lower()) if len(w) > 2]
        # The task's own punctuation first, then constants learned from documentation; plus output
        # lengths as candidate widths (e.g. zfill). Capped to keep the search bounded.
        own = sorted({c for ex in task['examples'] for s in ex if isinstance(s, str) for c in s if not c.isalnum()})
        consts = list(dict.fromkeys(own + self.constants))[:10]
        widths = sorted({len(y) for _, y in task['examples'] if isinstance(y, str)})[:3]
        out = []
        for s in self.skills:
            m, req = s['method'], s.get('required', 0)
            arg_sets = [()] if req == 0 else []
            takes_args = bool(s.get('params'))  # documented signature has parameters
            if takes_args and req <= 1 and m not in ('join',):
                arg_sets += [(c,) for c in consts]
                if m in ('zfill', 'center', 'ljust', 'rjust'):
                    arg_sets += [(w,) for w in widths]
            if req == 2 or m == 'replace':
                arg_sets += [(a, b) for a in consts for b in consts + [''] if a != b]
            r = self.relevance(s, goal)
            for args in arg_sets:
                out.append((-r, m, args))
        if any(s['method'] == 'join' for s in self.skills):
            r = self.relevance(next(s for s in self.skills if s['method'] == 'join'), goal)
            out += [(-r, 'JOIN', (c,)) for c in consts]
        out.sort(key=lambda t: (t[0], t[1], repr(t[2])) if self.use_docs or self.priors else (t[1], repr(t[2])))
        return out

    @staticmethod
    def expression(program):
        cur = 'x'
        for _, m, args in program:
            cur = f'{args[0]!r}.join({cur})' if m == 'JOIN' else f'{cur}.{m}({", ".join(map(repr, args))})'
        return cur

    def solve(self, task, budget_limit=20000, depth=2, probe_inputs=('', ' Ab cD ', 'a-b,c/d', 'X')):
        """One-step programs first, then two-step programs in order of combined step rank
        (lazy k-best pairs over all steps). After a fit, a short look-ahead checks for a rival
        fit that behaves differently on probe inputs: if one exists with equal relevance,
        the answer is 'not enough evidence'."""
        steps = self.steps(task); budget = Budget(budget_limit); n = len(steps)

        def fits(idx):
            program = [steps[i] for i in idx]; expr = self.expression(program); budget.spend()
            ok = all(run(expr, {'x': x}) == ('ok', y) for x, y in task['examples'])
            return (expr, program) if ok else None

        def candidates():
            for i in range(n):
                yield (i,)
            if depth >= 2:
                heap, seen = [(0, (0, 0))], {(0, 0)}
                while heap:
                    r, (i, j) = heapq.heappop(heap); yield (i, j)
                    for nxt in ((i + 1, j), (i, j + 1)):
                        if max(nxt) < n and nxt not in seen:
                            seen.add(nxt); heapq.heappush(heap, (nxt[0] + nxt[1], nxt))
        found = None
        try:
            gen = candidates()
            for idx in gen:
                hit = fits(idx)
                if hit and found is None:
                    found = (idx, hit); horizon = budget.used + 200
                elif hit and found:
                    a, b = found[1][0], hit[0]
                    differs = any(run(a, {'x': p}) != run(b, {'x': p}) for p in probe_inputs)
                    same_rank = sum(-steps[i][0] for i in found[0]) == sum(-steps[i][0] for i in idx)
                    if differs and same_rank and len(idx) == len(found[0]):
                        return {'program': None, 'attempts': budget.used, 'answer': 'not enough evidence',
                                'rivals': [a, b]}
                if found and budget.used >= horizon:
                    break
        except BudgetExhausted:
            pass
        if found:
            expr, program = found[1]
            return {'program': expr, 'methods': [m for _, m, _ in program], 'attempts': budget.used}
        return {'program': None, 'attempts': budget.used, 'answer': 'not enough evidence'}

    def learn_from(self, task, solution):
        """Procedural memory: associate goal words with the methods that verifiably solved a train task."""
        for g in {stem(w) for w in WORD.findall(task['goal'].lower()) if len(w) > 2}:
            for m in solution['methods']:
                self.priors[(g, m)] = self.priors.get((g, m), 0) + 1
