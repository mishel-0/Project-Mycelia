# Phase 0 — Fair evaluation harness

**Gate: PASS.** `tools/matched_readout_comparison.py` applies one Training-only RBF-SVM protocol (C selected on stratified Training validation) to every representation, on the deduplicated OMNIA split (5,410 train / 1,468 test), with McNemar paired tests.

| Representation (same SVM readout) | Test accuracy |
|---|---:|
| HOG + intensity | 89.58% |
| MYCELIA 1,024-colony features | 79.16% |
| MYCELIA + HOG concatenated | 85.69% (worse than HOG, χ²≈26) |

Caveat: the frozen MYCELIA memory saw validation images unlabeled during representation learning; Testing was untouched.
