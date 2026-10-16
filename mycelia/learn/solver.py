"""Typed program search over verified skills.

Steps: str method `x.m(args)`, function call `f(x)`, `MAP(step)` over a list,
`JOIN(sep)`. Every step runs through the restricted sandbox. Search order:
all 1-step programs, then 2- and 3-step programs by combined rank, where a
step's rank comes from the relevance of the task's goal words to the skill's
documented words (name matches count double) plus priors from solved
training tasks. A program must reproduce every visible example; if a rival
program with the same rank also fits but behaves differently on probes, the
answer is 'not enough evidence'.
"""
from __future__ import annotations
import heapq, re
from mycelia.pure.dsl import Budget, BudgetExhausted
from mycelia.pure.sandbox import run

WORD = re.compile(r'[a-z]+')
PROBES = {str: ['', ' Ab cD ', 'a,b c,d'], list: [[1, 2], [5, 1, 4], ['b', 'a']], int: [0, 9], float: [2.5]}


def stem(w):
    return w[:4]


def match(goal_word, word):
    """Shared beginning of at least 3 letters, covering the shorter word up to 4 letters:
    length~len, swap~swapcase, capital~capitalize."""
    k = min(4, len(goal_word), len(word))
    return k >= 3 and goal_word[:k] == word[:k]


def apply(step, v):
    kind = step[0]
    if kind == 'method':
        return run(f'x.{step[1][4:]}({", ".join(map(repr, step[2]))})', {'x': v})
    if kind == 'call':
        return run(f'{step[1]}(x{"".join(", " + repr(a) for a in step[2])})', {'x': v})
    if kind == 'map':
        if not isinstance(v, list):
            return ('error', 'TypeError')
        out = []
        for e in v:
            r = apply(step[1], e)
            if r[0] != 'ok':
                return r
            out.append(r[1])
        return ('ok', out)
    if kind == 'join':
        return run(f'{step[1]!r}.join(x)', {'x': v})
    raise ValueError(step)


def execute(program, x):
    v = ('ok', x)
    for step in program:
        v = apply(step, v[1])
        if v[0] != 'ok':
            return v
    return v


def show(program):
    cur = 'x'
    for s in program:
        if s[0] == 'method':
            cur = f'{cur}.{s[1][4:]}({", ".join(map(repr, s[2]))})'
        elif s[0] == 'call':
            cur = f'{s[1]}({cur}{"".join(", " + repr(a) for a in s[2])})'
        elif s[0] == 'map':
            inner = show([s[1]]).replace('x', 'e', 1)
            cur = f'[{inner} for e in {cur}]'
        else:
            cur = f'{s[1]!r}.join({cur})'
    return cur


def ops(program):
    return [s[1][1] if s[0] == 'map' else ('str.join' if s[0] == 'join' else s[1]) for s in program]


class Solver:
    def __init__(self, skills, constants=(), use_docs=True, priors=None, budget=10000, lookahead=150):
        self.skills = dict(skills); self.constants = list(constants); self.use_docs = use_docs
        self.priors = priors if priors is not None else {}; self.budget, self.lookahead = budget, lookahead

    def relevance(self, op, goal):
        s = self.skills.get(op)
        if s is None:
            return 0.
        score = 0.
        if self.use_docs:
            names = WORD.findall(op.split('.')[-1]); words = s.get('words', [])
            score += sum(2 * any(match(g, w) for w in names) + any(match(g, w) for w in words) for g in goal)
        return score + sum(2 * self.priors.get((stem(g), op), 0) for g in goal)

    def steps(self, task):
        goal = [w for w in WORD.findall(task['goal'].lower()) if len(w) > 2]
        own = sorted({ch for ex in task['examples'] for v in ex if isinstance(v, str) for ch in v if not ch.isalnum() and ch != ' '})
        consts = list(dict.fromkeys(own + [' '] + self.constants))[:8]
        out = []
        for op, s in self.skills.items():
            r = self.relevance(op, goal); has_params = bool(s.get('params'))
            if s['kind'] == 'method':
                if op == 'str.join':
                    out += [(r, ('join', c)) for c in consts]; continue
                if s['required'] == 0:
                    out.append((r, ('method', op, []))); out.append((r - .5, ('map', ('method', op, []))))
                if has_params and s['required'] <= 1:
                    out += [(r, ('method', op, [c])) for c in consts]
                if s['required'] == 2:
                    out += [(r, ('method', op, [a, b])) for a in consts for b in consts if a != b]
            else:
                if s['required'] == 0:
                    out.append((r, ('call', op, []))); out.append((r - .5, ('map', ('call', op, []))))
        # Higher relevance first; among equals, simpler steps (fewer arguments) first.
        complexity = lambda st: len(st[2]) if st[0] in ('method', 'call') else (len(st[1][2]) + 1 if st[0] == 'map' else 1)
        if self.use_docs or self.priors:
            out.sort(key=lambda t: (-t[0], complexity(t[1]), repr(t[1])))
        else:
            out.sort(key=lambda t: (complexity(t[1]), repr(t[1])))
        return [s for _, s in out], [r for r, _ in out]

    def solve(self, task, step_penalty=3):
        steps, rel = self.steps(task); n = len(steps); budget = Budget(self.budget)
        cache = {}

        def value(idx, k):  # memoised execution of a program prefix on example k
            key = (idx, k)
            if key not in cache:
                prev = ('ok', task['examples'][k][0]) if len(idx) == 1 else value(idx[:-1], k)
                cache[key] = apply(steps[idx[-1]], prev[1]) if prev[0] == 'ok' else prev
            return cache[key]

        def fits(idx):
            budget.spend()
            return all(value(idx, k) == ('ok', y) for k, (_, y) in enumerate(task['examples']))

        def candidates():
            """All 1-step programs, then 2- and 3-step programs from one queue ordered by
            summed step rank plus a penalty per extra step."""
            for i in range(n):
                yield (i,)
            heap, seen = [(0, (0, 0)), (step_penalty, (0, 0, 0))], {(0, 0), (0, 0, 0)}
            while heap:
                _, idx = heapq.heappop(heap); yield idx
                for k in range(len(idx)):
                    nxt = idx[:k] + (idx[k] + 1,) + idx[k + 1:]
                    if nxt[k] < n and nxt not in seen:
                        seen.add(nxt); heapq.heappush(heap, (sum(nxt) + step_penalty * (len(nxt) - 2), nxt))
        cost = lambda prog: sum(len(st[2]) if st[0] in ('method', 'call') else 1 for st in prog)
        found = horizon = None
        try:
            for idx in candidates():
                if not fits(idx):
                    continue
                prog = [steps[i] for i in idx]
                if found is None:
                    found, horizon = (idx, prog), budget.used + self.lookahead
                elif (len(idx) == len(found[0]) and sum(rel[i] for i in idx) == sum(rel[i] for i in found[0])
                      and cost(prog) == cost(found[1])):
                    probes = PROBES.get(type(task['examples'][0][0]), [])
                    if any(execute(found[1], p) != execute(prog, p) for p in probes):
                        return {'program': None, 'answer': 'not enough evidence', 'attempts': budget.used,
                                'rivals': [show(found[1]), show(prog)]}
                if found and budget.used >= horizon:
                    break
        except BudgetExhausted:
            pass
        if found:
            return {'program': found[1], 'shown': show(found[1]), 'ops': ops(found[1]), 'attempts': budget.used}
        return {'program': None, 'answer': 'not enough evidence', 'attempts': budget.used}

    def learn_from(self, task, solution):
        for g in {stem(w) for w in WORD.findall(task['goal'].lower()) if len(w) > 2}:
            for op in solution['ops']:
                self.priors[(g, op)] = self.priors.get((g, op), 0) + 1


def score(solver, tasks, learn=False):
    rows = []
    for t in tasks:
        sol = solver.solve(t)
        if sol['program'] is None:
            outcome = 'abstained'
        else:
            outcome = 'correct' if all(execute(sol['program'], x) == ('ok', y) for x, y in t['hidden']) else 'wrong'
            if learn and outcome == 'correct':
                solver.learn_from(t, sol)
        rows.append({'id': t['id'], 'family': t['family'], 'solvable': t['solvable'], 'cross_domain': t['cross_domain'],
                     'novel': t['novel_combination'], 'outcome': outcome, 'program': sol.get('shown'), 'attempts': sol['attempts']})
    return rows
