"""Growing character-context colonies for text, mirroring the image colonies'
resource-gated recruitment/recycling (mycelia/visual_memory.py) but implemented
natively for 1D token sequences rather than reshaping text into image patches.

No neural network. A specialist is a context-keyed next-character count table.
New specialists are recruited only when local prediction error (surprise) on
the current context exceeds a threshold and growth capacity remains; specialists
untouched for a while are recycled when capacity is full. This is the
"architecture is not fixed in advance" property applied to text.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TextConfig:
    order: int = 3          # characters of context a specialist conditions on
    max_specialists: int = 2000
    novelty_threshold: float = 0.6   # surprise (1 - predicted prob) to grow
    idle_recycle_after: int = 5000   # steps unused before a specialist may be recycled
    smoothing: float = 0.5
    # Electrical-oscillation-style global signal (SELFO analogy): a network-
    # wide scalar that rises when ANY specialist is repeatedly surprised and
    # decays over time, temporarily lowering the growth threshold everywhere
    # -- "distant parts react faster" without local evidence of their own yet.
    stress_enabled: bool = False
    stress_decay: float = 0.98        # per-step multiplicative decay toward 0
    stress_gain: float = 1.0          # how much one surprise adds to stress
    stress_threshold_floor: float = 0.15  # minimum effective novelty_threshold under max stress

    def __post_init__(self):
        if self.order < 1 or self.max_specialists < 1:
            raise ValueError('order and max_specialists must be positive')
        if not 0 < self.novelty_threshold <= 1:
            raise ValueError('novelty_threshold must be in (0, 1]')
        if self.smoothing <= 0:
            raise ValueError('smoothing must be positive')
        if not 0 <= self.stress_decay <= 1 or self.stress_gain < 0:
            raise ValueError('stress_decay must be in [0,1] and stress_gain must be nonnegative')
        if not 0 <= self.stress_threshold_floor <= self.novelty_threshold:
            raise ValueError('stress_threshold_floor must be in [0, novelty_threshold]')


class TextMycelium:
    """A context specialist is created lazily; growth is resource-gated by
    `max_specialists`, not set in advance. Each specialist tracks character
    counts and total touches so predictions and recycling both use real,
    inspectable state."""

    def __init__(self, config=None, alphabet=None):
        self.config = config or TextConfig()
        self.alphabet = sorted(set(alphabet)) if alphabet else []
        self._index = {ch: i for i, ch in enumerate(self.alphabet)}
        self.specialists = {}   # context(str) -> {'counts': dict(char->count), 'total': int, 'last_seen': int}
        self.step = 0
        self.recruited = 0
        self.recycled = 0
        self.rejected_for_capacity = 0
        self.stress = 0.0
        self.stress_history = []

    def _char_index(self, ch):
        idx = self._index.get(ch)
        if idx is None:
            idx = len(self.alphabet)
            self.alphabet.append(ch)
            self._index[ch] = idx
        return idx

    def predict(self, window):
        """Return (predicted_char_or_None, prob_dict) using the longest known
        suffix of the last `order` characters (`window`); falls back to the
        global unigram distribution. `window` must already be truncated to
        at most `self.config.order` characters by the caller (fit/observe) so
        this never re-slices a growing prefix."""
        for k in range(min(len(window), self.config.order), -1, -1):
            key = window[len(window)-k:]
            node = self.specialists.get(key)
            if node is not None and node['total'] > 0:
                v = len(self.alphabet) or 1
                probs = {ch: (node['counts'].get(ch, 0)+self.config.smoothing) /
                         (node['total']+self.config.smoothing*v) for ch in self.alphabet}
                best = max(probs, key=probs.get) if probs else None
                return best, probs
        v = len(self.alphabet) or 1
        uniform = {ch: 1.0/v for ch in self.alphabet} if self.alphabet else {}
        return (max(uniform, key=uniform.get) if uniform else None), uniform

    def _recycle_one(self):
        if not self.specialists:
            return False
        victim = min(self.specialists, key=lambda k: self.specialists[k]['last_seen'])
        if self.step - self.specialists[victim]['last_seen'] < self.config.idle_recycle_after:
            return False
        del self.specialists[victim]
        self.recycled += 1
        return True

    def observe(self, window, actual_char):
        """Update counts for the matched context; grow a new, more specific
        specialist only when the current prediction was surprising enough.
        `window` is the last `order` characters only (see `predict`)."""
        self._char_index(actual_char)
        for ch in window:
            self._char_index(ch)
        key = window
        _, probs = self.predict(window)
        surprise = 1.0 - probs.get(actual_char, 0.0)
        # Global broadcast: any specialist's surprise raises network-wide
        # stress; stress decays each step regardless of where it originated,
        # so a spike at one context can lower everyone's growth threshold
        # for a while -- the SELFO analogy, not local evidence sharing.
        if self.config.stress_enabled:
            # Bounded EMA of recent surprise, not an unbounded accumulator:
            # stress in [0,1] tracks "how confused has the network been
            # lately," spiking after a style/vocabulary shift and decaying
            # back down once specialists adapt to the new regime.
            self.stress = self.stress*self.config.stress_decay + (1-self.config.stress_decay)*surprise
            self.stress_history.append(self.stress)
            effective_threshold = max(self.config.stress_threshold_floor,
                                       self.config.novelty_threshold - self.config.stress_gain*self.stress)
        else:
            effective_threshold = self.config.novelty_threshold
        node = self.specialists.get(key)
        if node is None:
            if surprise >= effective_threshold:
                if len(self.specialists) >= self.config.max_specialists:
                    if not self._recycle_one():
                        self.rejected_for_capacity += 1
                        node = None
                    else:
                        node = {'counts': {}, 'total': 0, 'last_seen': self.step}
                        self.specialists[key] = node
                        self.recruited += 1
                else:
                    node = {'counts': {}, 'total': 0, 'last_seen': self.step}
                    self.specialists[key] = node
                    self.recruited += 1
        if node is not None:
            node['counts'][actual_char] = node['counts'].get(actual_char, 0) + 1
            node['total'] += 1
            node['last_seen'] = self.step
        self.step += 1
        return surprise

    def fit(self, text):
        order = self.config.order
        window = ''
        for ch in text:
            self.observe(window, ch)
            window = (window+ch)[-order:]

    def cross_entropy_bits(self, text):
        """Average -log2(p(actual)) over the text, without any further learning."""
        order = self.config.order
        window = ''
        total = 0.0
        for ch in text:
            _, probs = self.predict(window)
            p = max(probs.get(ch, 1e-9), 1e-9)
            total += -math.log2(p)
            window = (window+ch)[-order:]
        return total/len(text) if text else 0.0

    def summary(self):
        return {'step': self.step, 'specialists': len(self.specialists),
                'recruited': self.recruited, 'recycled': self.recycled,
                'rejected_for_capacity': self.rejected_for_capacity,
                'alphabet_size': len(self.alphabet)}


class FixedOrderNgram:
    """Trivial non-adaptive baseline: a single fixed-order n-gram table, no
    growth, no recycling. The 'HOG' of this domain."""

    def __init__(self, order=3, smoothing=0.5):
        self.order = order
        self.smoothing = smoothing
        self.counts = {}
        self.alphabet = set()

    def fit(self, text):
        self.alphabet.update(text)
        for i in range(len(text)):
            key = text[max(0, i-self.order):i]
            node = self.counts.setdefault(key, {})
            node[text[i]] = node.get(text[i], 0) + 1

    def predict_probs(self, context):
        key = context[-self.order:]
        node = self.counts.get(key, {})
        total = sum(node.values())
        v = len(self.alphabet) or 1
        return {ch: (node.get(ch, 0)+self.smoothing)/(total+self.smoothing*v) for ch in self.alphabet}

    def cross_entropy_bits(self, text):
        window = ''
        total = 0.0
        for ch in text:
            probs = self.predict_probs(window)
            p = max(probs.get(ch, 1e-9), 1e-9)
            total += -math.log2(p)
            window = (window+ch)[-self.order:]
        return total/len(text) if text else 0.0
