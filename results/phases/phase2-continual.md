# Phase 2 — Continual learning on OMNIA brain-MRI stream

**Gate: FAIL, decisively.** Target was ≤2 pts forgetting, within 5 pts of retrain-from-scratch, at ≥5x less cumulative compute. None of the three held.

## Final accuracy (8 chunks, all 5,410 train images streamed, 1,468 held-out test)

| Arm | IID stream | Class-incremental stream |
|---|---:|---:|
| MYCELIA running-centroid (stale features) | 65.33% | 62.13% |
| MYCELIA running-centroid (refreshed features) | 65.67% | 65.87% |
| HOG running-centroid (no retrain) | 67.64% | 67.64% |
| **HOG + SVM, retrained from scratch each chunk** | **89.58%** | **89.58%** |

Gap to retrain-from-scratch: **~24 points**, not the targeted 5.

## Cumulative compute (8 chunks)

| Arm | Seconds |
|---|---:|
| MYCELIA unlabeled learning | 600.7 |
| MYCELIA centroid refresh | 9.4 |
| **HOG + SVM retrain from scratch, every chunk** | **6.5** |

MYCELIA is **~92x slower** than simply retraining a conventional SVM from scratch on all data seen so far — the opposite of the "efficient because it doesn't retrain" hypothesis. Even just MYCELIA's centroid refresh (9.4s) is slower than HOG's full from-scratch SVM retrain (6.5s), because MYCELIA's feature vectors are 4,144-dimensional and retraining still means rescanning the colony bank.

## Forgetting

Class-incremental `mycelia_stale` starts at 1.0 (trivial — only 1 class's test images exist yet), then falls to 0.621 by the end. This is confounded with more classes entering the test pool each chunk, not a clean forgetting measurement — but there is no evidence of graceful, low-forgetting adaptation either.

## Conclusion

The central "symbiotic" bet of this plan — that incremental MYCELIA learning would stay competitive while avoiding the cost of retraining — did not pan out. Retraining a simple conventional classifier from scratch every chunk is both **more accurate and far cheaper** than MYCELIA's continual-learning path, on this dataset and this implementation. This is a real negative result, not a tuning artifact: MYCELIA's own learning step (`model.learn()`) is the dominant cost (600s) and still underperforms.

**Implication for the plan:** Phase 2 as scoped does not support claiming continual learning as MYCELIA's niche. Before any further "symbiotic" claims, the architecture needs either (a) a dramatically faster learning step, or (b) a setting where retraining from scratch is genuinely impossible (e.g., data deleted after each chunk / privacy constraints) — accuracy and speed alone do not favor it here.
