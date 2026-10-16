# Phase 11: swarm vision, pod-scale benchmark (2026-10-16)

Script: `tools/swarm_vision_pod.py` (run on a Runpod cpu3g 32 vCPU pod, 350 s, about $0.45 in total across two pods).
Task: a drone flies to a goal on unseen 24x24 maps using only what it perceives from 16x16 synthetic camera
patches (floor / wall / hazard). Entering a hazard fails. 10 seeds x 100 maps per cell. Navigation rule is fixed;
only the perception method differs. All methods get identical labelled patches.
Methods: `mycelia` (unsupervised receptor field + teach-only trace memory), `cnn_tuned` (wider CNN, augmentation,
training length chosen on held-out labelled data), `cnn_basic`, `raw_svm`.
Ground-truth navigation ceiling is about 0.9 (the trail rule fails some maps with perfect sight).

## Task success, MYCELIA vs tuned CNN (95% paired-bootstrap interval over seeds on the difference)

| noise | labels/class | MYCELIA | cnn_tuned | difference (95%) |
|---|---|---|---|---|
| 0.05 | 5 | 0.805 | 0.488 | +0.317 (+0.116, +0.513) |
| 0.05 | 10 | 0.806 | 0.622 | +0.184 (+0.059, +0.336) |
| 0.05 | 20 | 0.839 | 0.641 | +0.198 (+0.078, +0.327) |
| 0.05 | 40 | 0.834 | 0.431 | +0.403 (+0.163, +0.648) |
| 0.05 | 80 | 0.832 | 0.801 | +0.031 (-0.005, +0.078) tie |
| 0.05 | 150 | 0.837 | 0.822 | +0.015 (-0.004, +0.047) tie |
| 0.15 | 5 | 0.232 | 0.202 | +0.030 (-0.123, +0.175) tie |
| 0.15 | 10 | 0.430 | 0.313 | +0.117 (-0.112, +0.360) tie |
| 0.15 | 20 | 0.553 | 0.517 | +0.036 (-0.185, +0.275) tie |
| 0.15 | 40 | 0.706 | 0.545 | +0.161 (-0.004, +0.361) tie |
| 0.15 | 80 | 0.704 | 0.488 | +0.216 (+0.084, +0.353) MYCELIA |
| 0.15 | 150 | 0.731 | 0.549 | +0.182 (-0.015, +0.393) tie |
| 0.30 | 5 | 0.004 | 0.072 | -0.068 (-0.137, -0.008) CNN |
| 0.30 | 10 | 0.002 | 0.133 | -0.131 (-0.227, -0.055) CNN |
| 0.30 | 20 | 0.001 | 0.165 | -0.164 (-0.291, -0.053) CNN |
| 0.30 | 40 | 0.013 | 0.081 | -0.068 (-0.115, -0.024) CNN |
| 0.30 | 80 | 0.022 | 0.030 | -0.008 (-0.038, +0.015) tie |
| 0.30 | 150 | 0.057 | 0.095 | -0.038 (-0.088, +0.005) tie |

## Speed (single process, ms per cell, numbers stored)
mycelia 0.31 ms / 123,600; cnn_tuned 0.10 ms / 37,827; cnn_basic 0.027 ms / 9,571; raw_svm 0.017 ms / 30,208.
MYCELIA is slower and larger than the CNNs at this size.

## Reading
- MYCELIA's lead holds against a fairly trained CNN in one regime: clean images and 5-40 labels per class.
- It ties with plenty of labels (80-150) and loses at heavy noise.
- The compute-efficiency claim is not supported here.
- Synthetic textures drawn by the author; no real images yet. Research simulation only.
