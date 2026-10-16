# Phase 3 — Few-label efficiency vs. HOG

**Gate: FAIL.** MYCELIA never beats HOG+same-readout at any label count tested (1 to 320 labels/class), with the same nearest-centroid classifier on both representations.

| Labels/class | MYCELIA | HOG | Difference |
|---:|---:|---:|---:|
| 1 | 39.11% | 42.74% | −3.63 |
| 5 | 50.13% | 56.06% | −5.93 |
| 20 | 58.14% | 62.48% | −4.34 |
| 40 | 61.82% | 64.32% | −2.50 |
| 160 | 64.95% | 65.71% | −0.76 |
| 320 | 65.18% | 65.73% | −0.55 |

The gap shrinks as labels increase (consistent with Phase 0/1: MYCELIA's representation is simply weaker, not more label-efficient) but never crosses zero. HOG wins at every single point, including with just 1 label per class.

## Conclusion
Three of the four capability bets in this plan (accuracy parity, continual learning, few-label learning) have now failed against a trivial conventional baseline. Only Phase 4 (masked-content prediction) has shown a genuine, reproducible MYCELIA-specific strength. Phase 8 builds directly on that one proven capability rather than re-testing accuracy-based claims.
