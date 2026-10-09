"""Phase 9 gate: does electrical-oscillation-style global stress signaling
(mycelia/text_memory.py, TextConfig.stress_enabled) help MYCELIA adapt faster
to a genuinely novel book it was never trained on, vs. (a) a trivial fixed
n-gram baseline and (b) the plain grower from Phase 8 (no stress signal)?

This is a true domain-shift test, not a random split: training excludes the
held-out book entirely, so vocabulary/style at test time is genuinely new.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mycelia.text_memory import TextMycelium, FixedOrderNgram, TextConfig


def per_position_bits(model, text, order):
    """Frozen scoring: predict only, never update. Used to confirm a model
    that never sees the held-out book at all can't adapt to it (sanity
    baseline for the online test below)."""
    bits = np.empty(len(text))
    window = ''
    is_mycelia = hasattr(model, 'specialists')
    for i, ch in enumerate(text):
        probs = model.predict(window)[1] if is_mycelia else model.predict_probs(window)
        p = max(probs.get(ch, 1e-9), 1e-9)
        bits[i] = -math.log2(p)
        window = (window+ch)[-order:]
    return bits


def prequential_bits(model, text, order):
    """Online/prequential: predict the next character, THEN let the model
    learn it, one character at a time. This is the only protocol in which a
    'react faster to a domain shift' mechanism can possibly show an effect --
    frozen scoring (above) gives it no opportunity to adapt at all."""
    bits = np.empty(len(text))
    window = ''
    is_mycelia = hasattr(model, 'specialists')
    for i, ch in enumerate(text):
        probs = model.predict(window)[1] if is_mycelia else model.predict_probs(window)
        p = max(probs.get(ch, 1e-9), 1e-9)
        bits[i] = -math.log2(p)
        if is_mycelia:
            model.observe(window, ch)
        else:
            model.alphabet.add(ch)
            node = model.counts.setdefault(window, {})
            node[ch] = node.get(ch, 0)+1
        window = (window+ch)[-order:]
    return bits


def paired_ci(a, b, seed, resamples=20000, chunk=200):
    delta = a-b
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for start in range(0, resamples, chunk):
        n = min(chunk, resamples-start)
        idx = rng.integers(0, len(delta), size=(n, len(delta)))
        boot[start:start+n] = delta[idx].mean(axis=1)
    return float(delta.mean()), [float(v) for v in np.quantile(boot, [.025, .975])], float((delta < 0).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-books', nargs='+', type=Path, required=True)
    p.add_argument('--held-out-book', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--order', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    train_text = ''.join(path.read_text(encoding='utf-8') for path in a.train_books)
    held_out = a.held_out_book.read_text(encoding='utf-8')

    unique_contexts = len(set(train_text[max(0, i-a.order):i] for i in range(len(train_text))))
    capacity = int(unique_contexts*1.2)  # train-only sizing, same discipline as Phase 8

    baseline = FixedOrderNgram(order=a.order); baseline.fit(train_text)
    plain = TextMycelium(TextConfig(order=a.order, max_specialists=capacity)); plain.fit(train_text)
    stressed = TextMycelium(TextConfig(order=a.order, max_specialists=capacity, stress_enabled=True,
                                       stress_gain=0.5, stress_decay=0.97, stress_threshold_floor=0.1))
    stressed.fit(train_text)

    frozen_baseline_bits = per_position_bits(baseline, held_out, a.order)
    frozen_plain_bits = per_position_bits(plain, held_out, a.order)
    frozen_stressed_bits = per_position_bits(stressed, held_out, a.order)

    # Online/prequential: each model continues learning from the held-out
    # book as it scores it, starting from its train-books state. This is the
    # only protocol where "reacts faster to a domain shift" can be tested.
    baseline_bits = prequential_bits(baseline, held_out, a.order)
    plain_bits = prequential_bits(plain, held_out, a.order)
    stressed_bits = prequential_bits(stressed, held_out, a.order)

    d_plain_base, ci_plain_base, frac_plain = paired_ci(plain_bits, baseline_bits, a.seed)
    d_stress_base, ci_stress_base, frac_stress_base = paired_ci(stressed_bits, baseline_bits, a.seed+1)
    d_stress_plain, ci_stress_plain, frac_stress_plain = paired_ci(stressed_bits, plain_bits, a.seed+2)

    # Does stress help most in the opening stretch of the held-out book,
    # before the plain grower would have adapted locally anyway?
    window = min(5000, len(held_out)//4) or 1
    early_stress, early_plain, early_baseline = (stressed_bits[:window].mean(),
                                                  plain_bits[:window].mean(), baseline_bits[:window].mean())

    result = {
        'schema': 'mycelia.phase9-electrical-signaling.v1', 'scope': __doc__.strip(),
        'train_books': [str(x) for x in a.train_books], 'held_out_book': str(a.held_out_book),
        'train_chars': len(train_text), 'held_out_chars': len(held_out),
        'order': a.order, 'capacity': capacity,
        'frozen_sanity_mean_bits_per_char': {'baseline': float(frozen_baseline_bits.mean()),
                               'plain_grower': float(frozen_plain_bits.mean()),
                               'stress_signaled': float(frozen_stressed_bits.mean())},
        'online_prequential_mean_bits_per_char': {'baseline': float(baseline_bits.mean()),
                               'plain_grower': float(plain_bits.mean()),
                               'stress_signaled': float(stressed_bits.mean())},
        'early_window_chars': window,
        'early_mean_bits_per_char': {'baseline': float(early_baseline),
                                     'plain_grower': float(early_plain),
                                     'stress_signaled': float(early_stress)},
        'plain_vs_baseline': {'delta': d_plain_base, 'ci': ci_plain_base, 'frac_better': frac_plain},
        'stress_vs_baseline': {'delta': d_stress_base, 'ci': ci_stress_base, 'frac_better': frac_stress_base},
        'stress_vs_plain': {'delta': d_stress_plain, 'ci': ci_stress_plain, 'frac_better': frac_stress_plain},
        'gate': 'stress_signaled beats BOTH baseline and plain_grower, CI excluding zero',
        'gate_passed': bool(ci_stress_base[1] < 0 and ci_stress_plain[1] < 0),
        'plain_summary': plain.summary(), 'stress_summary': stressed.summary(),
    }
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('online_prequential_mean_bits_per_char', 'early_mean_bits_per_char',
          'stress_vs_baseline', 'stress_vs_plain', 'gate_passed')}, indent=2))


if __name__ == '__main__':
    main()
