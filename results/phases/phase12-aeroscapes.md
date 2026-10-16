# Phase 12: AeroScapes real drone imagery (2026-10-16)

Script: `tools/aeroscapes_benchmark.py`, run on a Runpod cpu3g 32 vCPU pod (237 s). Data: AeroScapes (CC-BY-SA-4.0),
3,269 drone images, per-pixel masks. 16x16 patches, label = dominant mask class (>= 90% purity), 126,284 patches
from 121 drone sequences. Split BY SEQUENCE (no flight on both sides): 89,420 train / 36,864 test patches.
Classes with enough data: mask indices 0 (background), 8, 9, 10, 11 (five classes; chance = 20% balanced accuracy).
Test set: up to 300 patches per class. 10 seeds per label budget.

Methods (identical labels): mycelia (unsupervised receptors on 2,000 unlabelled patches + teach-only trace memory,
16x16 grayscale); mobilenet (ImageNet-pretrained MobileNetV3-small features, colour, + logistic regression);
cnn_scratch (small CNN with augmentation, grayscale); raw_svm (RBF-SVM on gray pixels).

## Stage 1: balanced accuracy vs labels per class

| labels/class | MYCELIA | MobileNet (pretrained) | CNN scratch | raw SVM | MYCELIA - MobileNet (95%) |
|---|---|---|---|---|---|
| 5 | 0.348 | 0.617 | 0.380 | 0.371 | -0.269 (-0.285, -0.251) |
| 10 | 0.368 | 0.643 | 0.419 | 0.401 | -0.276 (-0.293, -0.258) |
| 20 | 0.390 | 0.680 | 0.455 | 0.427 | -0.290 (-0.316, -0.270) |
| 40 | 0.417 | 0.687 | 0.446 | 0.444 | -0.270 (-0.291, -0.247) |
| 150 | 0.472 | 0.720 | 0.453 | 0.470 | -0.248 (-0.257, -0.240) |

Fit time (mean): MYCELIA 0.009 s, MobileNet probe 0.017 s, CNN scratch 2.0 s, raw SVM 0.033 s.

## Stage 2: synthetic low-light + noise shift, streamed (10 batches of 90 unlabelled tiles + 6 labelled)
Accuracy at batch 0 -> 10 (5 seeds): frozen MYCELIA 0.229 -> 0.229; MYCELIA memory-only 0.229 -> 0.238;
MYCELIA memory + receptor adaptation 0.229 -> 0.227; MobileNet probe refit 0.626 -> 0.645.

## Reading
- On real drone imagery MYCELIA's gray-patch receptor vision is far behind a pretrained MobileNet at every label
  budget (about 25-29 points) and only matches the other from-scratch baselines at 150 labels, so the few-label
  advantage seen on synthetic textures did NOT transfer.
- Adaptation under shift did not help here (0.229 -> 0.227-0.238).
- Caveats: patches are small (16x16 gray) and give MYCELIA little to work with; MobileNet sees colour and pretraining
  on ImageNet. A fair next step is giving MYCELIA colour and larger patches, or pretrained-free baselines only.
- Research benchmark, not flight software.
