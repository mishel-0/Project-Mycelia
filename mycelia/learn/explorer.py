"""Choosing what to read next from a catalog of documentation files.

goal-driven: score files by overlap between unsolved *training* task goals and
             the file name plus the sentences in already-read documents that
             link to that file.
curiosity:   score files by unresolved references from already-read documents
             (how many distinct documents point to it), plus unverified claims
             it could resolve; ignores tasks entirely.
random / alphabetical baselines. Held-out evaluation tasks are never visible.
"""
from __future__ import annotations
import random, re
from collections import Counter, defaultdict
from .solver import match, WORD

ROLE = re.compile(r':(?:mod|func|meth|class|data):`~?([\w.]+)')


def module_file(catalog):
    """Map module/top-level names to catalog files (math -> Doc/library/math.rst)."""
    out = {}
    for f in catalog:
        stem = f.rsplit('/', 1)[-1].removesuffix('.rst')
        out[stem] = f
    # Built-in functions (len, sorted, ...) are documented in Doc/builtins/functions.rst; the file
    # Doc/library/builtins.rst describes the `builtins` module object instead (bug found in the first run).
    functions = next((f for f in catalog if f.endswith('builtins/functions.rst')), None)
    if functions:
        out['builtins'] = functions
    return out


def references(text, catalog):
    files = module_file(catalog); refs = Counter(); context = defaultdict(list)
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        for r in ROLE.findall(sentence):
            head = r.split('.')[0]
            target = files.get(head) or (files.get('builtins') if '.' not in r and head in BUILTIN_NAMES else None)
            if target:
                refs[target] += 1; context[target].append(sentence)
    return refs, context


BUILTIN_NAMES = {'len', 'sorted', 'sum', 'min', 'max', 'abs', 'round', 'int', 'float', 'str', 'list', 'map', 'filter', 'zip'}


class Explorer:
    def __init__(self, catalog, strategy, seed=0):
        self.catalog, self.strategy, self.rng = list(catalog), strategy, random.Random(seed)
        self.read, self.refs, self.context, self.unverified = [], Counter(), defaultdict(list), Counter()

    def observe(self, path, text, unverified_subjects=()):
        self.read.append(path); refs, ctx = references(text, self.catalog)
        for f, n in refs.items():
            self.refs[f] += 1  # distinct documents pointing to f
            self.context[f] += ctx[f]
        files = module_file(self.catalog)
        for subj in unverified_subjects:
            f = files.get(subj.split('.')[0])
            if f:
                self.unverified[f] += 1

    def next(self, unsolved_goals=()):
        options = [f for f in self.catalog if f not in self.read]
        if not options:
            return None
        if self.strategy == 'alphabetical':
            return sorted(options)[0]
        if self.strategy == 'random':
            return self.rng.choice(options)
        if self.strategy == 'curiosity':
            return max(options, key=lambda f: (self.refs[f] + .5 * self.unverified[f], -len(f), f))
        goal_words = [w for g in unsolved_goals for w in WORD.findall(g.lower()) if len(w) > 2]

        def score(f):
            words = WORD.findall(f.rsplit('/', 1)[-1].removesuffix('.rst').lower())
            words += [w for s in self.context[f] for w in WORD.findall(s.lower())]
            return sum(any(match(g, w) for w in words) for g in set(goal_words))
        return max(options, key=lambda f: (score(f), self.refs[f], f))
