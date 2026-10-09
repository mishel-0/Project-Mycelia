"""Phase 8 gate: does MYCELIA's growing character-colony predict held-out text
better than a trivial fixed-order n-gram baseline?

Metric: bits per character (cross-entropy), lower is better. Both models train
on the same prefix of the text and are scored, without further learning, on a
disjoint held-out suffix. Paired per-position bootstrap CI, same protocol as
Phase 4 (tools/masked_visual_prediction_test.py).
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
    """Bits for each position individually (for paired comparison), not just
    the mean). Uses a rolling `order`-character window, never re-slicing a
    growing prefix (that was quadratic and made 3.6MB infeasible)."""
    bits = np.empty(len(text))
    window = ''
    is_mycelia = hasattr(model, 'specialists')
    for i, ch in enumerate(text):
        probs = model.predict(window)[1] if is_mycelia else model.predict_probs(window)
        p = max(probs.get(ch, 1e-9), 1e-9)
        bits[i] = -math.log2(p)
        window = (window+ch)[-order:]
    return bits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-fraction', type=float, default=0.9)
    p.add_argument('--max-specialists', type=int, default=4000)
    p.add_argument('--order', type=int, default=4)
    p.add_argument('--novelty-threshold', type=float, default=0.6)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    text = a.corpus.read_text(encoding='utf-8')
    split = int(len(text)*a.train_fraction)
    train_text, held_out = text[:split], text[split:]

    mycelia = TextMycelium(TextConfig(order=a.order, max_specialists=a.max_specialists,
                                      novelty_threshold=a.novelty_threshold))
    mycelia.fit(train_text)
    baseline = FixedOrderNgram(order=a.order)
    baseline.fit(train_text)

    mycelia_bits = per_position_bits(mycelia, held_out, a.order)
    baseline_bits = per_position_bits(baseline, held_out, a.order)
    delta = mycelia_bits - baseline_bits  # negative = MYCELIA better
    rng = np.random.default_rng(a.seed)
    # Resample indices and sum in chunks instead of materializing a
    # (resamples x len(delta)) array, which can be many GB for long texts.
    resamples, chunk = 20000, 200
    boot = np.empty(resamples)
    for start in range(0, resamples, chunk):
        n = min(chunk, resamples-start)
        idx = rng.integers(0, len(delta), size=(n, len(delta)))
        boot[start:start+n] = delta[idx].mean(axis=1)
    ci = [float(v) for v in np.quantile(boot, [.025, .975])]

    result = {
        'schema': 'mycelia.phase8-text-growth.v1', 'scope': __doc__.strip(),
        'corpus': str(a.corpus), 'train_chars': split, 'held_out_chars': len(held_out),
        'order': a.order, 'max_specialists': a.max_specialists,
        'novelty_threshold': a.novelty_threshold, 'seed': a.seed,
        'mycelia_summary': mycelia.summary(),
        'mycelia_mean_bits_per_char': float(mycelia_bits.mean()),
        'baseline_mean_bits_per_char': float(baseline_bits.mean()),
        'mean_delta_mycelia_minus_baseline': float(delta.mean()),
        'paired_bootstrap_95_ci': ci,
        'fraction_positions_mycelia_better': float((delta < 0).mean()),
        'gate': 'mycelia beats fixed n-gram baseline (negative delta) with 95% CI excluding zero',
        'gate_passed': bool(ci[1] < 0),
    }
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('mycelia_mean_bits_per_char',
          'baseline_mean_bits_per_char', 'mean_delta_mycelia_minus_baseline',
          'paired_bootstrap_95_ci', 'fraction_positions_mycelia_better', 'gate_passed')}, indent=2))
    print(json.dumps(result['mycelia_summary'], indent=2))


if __name__ == '__main__':
    main()
