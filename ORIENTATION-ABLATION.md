# MYCELIA 64px spatial/orientation ablation

This preliminary experiment used only the original **Training** partition. It
decoded no Testing image and read no Testing label. From exact-deduplicated
Training data, 300 images (75/class) fit each unlabeled MYCELIA representation;
a separate 400 images (100/class) evaluated fixed readouts. The seed was
20261008. Every configuration used one pass, 8×8 local patches, fixed
full-frame preprocessing, multiscale cues, and the same labels/readouts.

The new spatial-pyramid representation preserves colony assignments at global,
2×2, 4×4, and—at 64px—8×8 spatial levels. It contains no labels, image IDs, or
Testing information. The fixed RBF-SVM is a diagnostic readout that tests
whether the frozen representation contains separable class information; its
contribution is reported separately.

| Unlabeled representation | Colonies | Feature width | Fit time | Centroid balanced accuracy | Fixed-SVM balanced accuracy | Cue reconstruction MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32px edge magnitude + texture | 256 | 5,424 | 20.08 s | 64.75% | 71.75% | 0.00845 |
| 32px signed x/y gradients | 256 | 5,424 | 17.94 s | 64.00% | 68.50% | 0.00536 |
| **64px edge magnitude + texture** | **256** | **21,952** | **54.13 s** | **69.75%** | **76.75%** | 0.00794 |
| 64px signed x/y gradients | 256 | 21,952 | 42.77 s | 67.75% | 74.75% | **0.00489** |
| 64px x/y edge energy | 256 | 21,952 | 46.80 s | 70.25% | 76.25% | 0.00751 |
| 64px edge magnitude + texture | 512 | 43,712 | 63.58 s | 70.00% | 76.00% | 0.00748 |

The matched conventional 32px HOG/intensity representation scored **80.0%**
with the same 300 labels and fixed SVM. The best MYCELIA result therefore
remains 3.25 points behind on this Training-validation probe.

Feature fusion did not close the gap:

| Fusion | Fixed-SVM balanced accuracy |
| --- | ---: |
| 32px texture + signed gradient colonies | 73.00% |
| 64px texture + signed gradient colonies | 76.00% |
| 64px texture + axis-energy colonies | 76.25% |
| 64px all three receptor populations | 76.75% |
| 32px + 64px texture colonies | **77.25%** |
| All resolutions/receptor populations | 76.75% |

Doubling capacity reduced novelty overflow from 4,429 to 3,835 of 19,200 patch
exposures and improved reconstruction MSE from 0.00794 to 0.00748, but did not
improve class separation. More colonies are therefore not the next lever.
Signed orientation improved reconstruction but hurt classification; polarity
is not the missing signal. Multi-resolution detail produced the only consistent
classification gain.

The next representation must change the learning objective, not merely the
capacity or receptor combination. The candidate is a phase-tolerant local
frequency/shape representation trained to predict genuinely masked neighboring
regions and cross-sequence 3D patches. Admission requires beating the 80.0%
matched HOG control on a predeclared Training-validation probe before any fresh
test cohort is opened.

These results are architecture-selection evidence only. They do not update the
66.76% saved Testing result, establish patient-level generalization, or support
clinical diagnosis.
