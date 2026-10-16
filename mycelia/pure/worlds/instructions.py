"""dax/blick/wug world: invented words for actions, modifiers and connectives.

Each episode draws a fresh random vocabulary and hidden rules. Demonstrations
show phrase -> action sequence; test phrases are combinations never shown
(a modifier on a primitive it was never seen with, and composites of those).
One primitive can be shown only inside compositions, never alone.
"""
from __future__ import annotations
import random
from ..dsl import UNARY, BINARY, apply_unary, apply_binary

SYLLABLES = ['da', 'x', 'bl', 'ick', 'wu', 'g', 'fe', 'p', 'zo', 'r', 'ki', 'm', 'lu', 'n', 'po', 't', 'ra', 's', 'ne', 'v']
ACTIONS = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'BLACK', 'WHITE', 'ORANGE', 'PINK']
# Hidden operators are drawn non-uniformly, so learned search priors can help.
UNARY_WEIGHTS = {('repeat', 2): 5, ('repeat', 3): 5, ('mirror',): 3, ('reverse',): 1, ('repeat', 4): 2}
BINARY_WEIGHTS = {('concat_xy',): 4, ('concat_yx',): 3, ('around',): 2, ('interleave',): 1}


def make_word(rng, used):
    while True:
        w = ''.join(rng.choice(SYLLABLES) for _ in range(2))
        if len(w) >= 3 and w not in used:
            used.add(w); return w


class Episode:
    def __init__(self, seed, n_primitives=4, hide_alone=True, noise=False, demos_per_modifier=2, binary_demos=2):
        rng = random.Random(seed); used = set()
        self.seed = seed
        acts = rng.sample(ACTIONS, 2 * n_primitives)  # each primitive is a two-action sequence
        self.prims = {make_word(rng, used): acts[2 * i:2 * i + 2] for i in range(n_primitives)}
        u = self._draw(rng, UNARY_WEIGHTS, 2)
        b = self._draw(rng, BINARY_WEIGHTS, 2)
        self.unary = {make_word(rng, used): op for op in u}; self.binary = {make_word(rng, used): op for op in b}
        words = list(self.prims)
        self.hidden_alone = words[-1] if hide_alone else None
        demos = [([w], self.prims[w]) for w in words if w != self.hidden_alone]
        self.held_out_pairs = set()
        for m in self.unary:
            seen = rng.sample(words, demos_per_modifier)
            if self.hidden_alone and self.hidden_alone not in seen and m == next(iter(self.unary)):
                seen[0] = self.hidden_alone  # the hidden primitive appears only with a modifier
            demos += [([p, m], self.run([p, m])) for p in seen]
            self.held_out_pairs |= {(p, m) for p in words if p not in seen}
        for c in self.binary:
            for _ in range(binary_demos):
                x, y = rng.sample([w for w in words if w != self.hidden_alone], 2)
                demos.append(([x, c, y], self.run([x, c, y])))
        self.noisy_index = None
        if noise:  # corrupt one demonstration of a modifier
            i = rng.choice([i for i, (p, _) in enumerate(demos) if len(p) == 2])
            out = list(demos[i][1]); out[-1] = next(a for a in ACTIONS if a != out[-1]); demos[i] = (demos[i][0], out)
            self.noisy_index = i
        self.demos = demos
        seen_phrases = {tuple(p) for p, _ in demos}
        tests = [[p, m] for p, m in sorted(self.held_out_pairs)]
        for c in self.binary:
            for (p, m) in sorted(self.held_out_pairs)[:3]:
                q = rng.choice(words); tests.append([p, m, c, q])
        self.tests = [(t, self.run(t)) for t in tests if tuple(t) not in seen_phrases]

    @staticmethod
    def _draw(rng, weights, k):
        pool = dict(weights); out = []
        for _ in range(k):
            ops = list(pool); op = rng.choices(ops, [pool[o] for o in ops])[0]; out.append(op); pool.pop(op)
        return out

    def term(self, words):
        x = list(self.prims[words[0]])
        return apply_unary(self.unary[words[1]], x) if len(words) > 1 else x

    def run(self, phrase):
        """Ground-truth semantics: term [binary term]; term = primitive [unary]."""
        for i, w in enumerate(phrase):
            if w in self.binary:
                return apply_binary(self.binary[w], self.term(phrase[:i]), self.term(phrase[i + 1:]))
        return self.term(phrase)
