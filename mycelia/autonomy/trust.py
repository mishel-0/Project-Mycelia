"""Per-sender trust from evidence: Beta(successes + 1, failures + 1)."""
from __future__ import annotations


class Trust:
    def __init__(self):
        self.counts = {}

    def update(self, sender, ok):
        s, f = self.counts.get(sender, (0, 0)); self.counts[sender] = (s + bool(ok), f + (not ok))

    def mean(self, sender):
        s, f = self.counts.get(sender, (0, 0)); return (s + 1) / (s + f + 2)

    def n(self, sender):
        return sum(self.counts.get(sender, (0, 0)))

    def reject(self, sender):      # enough evidence that this sender's knowledge fails here
        return self.n(sender) >= 3 and self.mean(sender) < .35

    def skip_verify(self, sender):  # enough evidence that this sender's knowledge holds here
        return self.n(sender) >= 5 and self.mean(sender) > .85
