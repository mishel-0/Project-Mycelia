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
    return w[:5]


class Solver:
    def __init__(self, skills, constants=(), use_docs=True, priors=None):
        self.skills = [s for s in skills if s.get('status', 'verified') in ('verified', 'observed', 'unverified', 'contested')]
        self.constants, self.use_docs = list(constants), use_docs
        self.priors = priors if priors is not None else {}

    def relevance(self, skill, goal_words):
        score = 0.
        if self.use_docs:
            words = {stem(w) for w in skill.get('words', [])} | {stem(skill['method'])}
            score += sum(stem(g) in words for g in goal_words)
        for g in goal_words:
            score += 2 * self.priors.get((stem(g), skill['method']), 0)
        return score

    def steps(self, task):
        goal = [w for w in WORD.findall(task['goal'].lower()) if len(w) > 2]
        consts = list(dict.fromkeys(self.constants + sorted({c for ex in task['examples'] for s in ex for c in str(s)
                                                             if not c.isalnum() and isinstance(s, str)})))[:8]
        out = []
        for s in self.skills:
            m, req = s['method'], s.get('required', 0)
            arg_sets = [()] if req == 0 else []
            if req <= 1 and m not in ('join',):
                arg_sets += [(c,) for c in consts]
            if req == 2 or m == 'replace':
                arg_sets += [(a, b) for a in consts for b in consts + [''] if a != b]
            r = self.relevance(s, goal)
            for args in arg_sets:
                out.append((-r, m, args))
        if any(s['method'] == 'join' for s in self.skills):
            r = self.relevance(next(s for s in self.skills if s['method'] == 'join'), goal)
            out += [(-r, 'JOIN', (c,)) for c in consts]
        out.sort(key=lambda t: (t[0], t[1], t[2]) if self.use_docs or self.priors else (t[1], t[2]))
        return out

    @staticmethod
    def expression(program):
        cur = 'x'
        for _, m, args in program:
            cur = f'{args[0]!r}.join({cur})' if m == 'JOIN' else f'{cur}.{m}({", ".join(map(repr, args))})'
        return cur

    def solve(self, task, budget_limit=20000, depth=3):
        steps = self.steps(task); budget = Budget(budget_limit); n = len(steps)
        # Best-first over programs: rank = sum of step ranks, shorter programs first at equal rank.
        heap = [(i, 1, (i,)) for i in range(n)]; heapq.heapify(heap); seen = set()
        try:
            while heap:
                rank, length, idx = heapq.heappop(heap)
                if idx in seen:
                    continue
                seen.add(idx); program = [steps[i] for i in idx]; expr = self.expression(program); budget.spend()
                if all(run(expr, {'x': x}) == ('ok', y) for x, y in task['examples']):
                    return {'program': expr, 'methods': [m for _, m, _ in program], 'attempts': budget.used}
                if length < depth:
                    for j in range(min(n, 60)):  # extend with the 60 most relevant steps
                        heapq.heappush(heap, (rank + j + 1, length + 1, idx + (j,)))
        except BudgetExhausted:
            pass
        return {'program': None, 'attempts': budget.used, 'answer': 'not enough evidence'}

    def learn_from(self, task, solution):
        """Procedural memory: associate goal words with the methods that verifiably solved a train task."""
        for g in {stem(w) for w in WORD.findall(task['goal'].lower()) if len(w) > 2}:
            for m in solution['methods']:
                self.priors[(g, m)] = self.priors.get((g, m), 0) + 1
