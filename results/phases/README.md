# Phase results (research log)

Each file below was produced by a script in `tools/` under a matched-baseline protocol: every representation is compared with the **same** readout, tuning uses Training data only, and uncertainty comes from paired bootstrap intervals. Failures are kept, not tuned away.

| Phase | Question | Outcome | Files |
|---|---|---|---|
| 0 | MYCELIA vs. HOG with an identical SVM readout (brain MRI, 1,468 clean test images) | HOG+SVM **89.58%**, MYCELIA 32px **79.16%**, combined 85.69% (worse than HOG alone) | `phase0-matched-readout.*` |
| 1 | Does 64px input close the gap? | MYCELIA **82.36%** vs. HOG 89.58%: gap shrank from ~10 to ~7 points, below the pre-set target | (run output, see Phase 0 protocol) |
| 2 | Continual learning vs. retraining an SVM each chunk | **Failed.** MYCELIA 62–66% in 600 s; retrained SVM 89.58% in 6.5 s | `phase2-continual.*` |
| 3 | Fewer labels needed than HOG? | **Failed** at every label count (1–320 per class) | `phase3-label-efficiency.*` |
| 4 | Predict masked image patches | **Passed** on 2 seeds (95% CI excludes zero, ~80% of images improved) | `tools/masked_visual_prediction_test.py` |
| 8 | Growing character-colony vs. fixed n-gram on 3.6 MB of public-domain books | Identical once capacity is sized to the data; worse when starved | `phase8-text*` |
| 9 | Global "stress" signal to react faster to a new book | **No effect** (delta 0.0); recruitment threshold is already saturated for novel text | `phase9-electrical-signaling.*` |
| 10 / Task 1 | Grown network vs. minimum spanning tree and complete graph (5 layouts) | Partial pass: efficiency within 8% of the MST, +16% random-damage survival, 39% longer backbone, weak under targeted cuts | `phase10-topology-task1.*` |

## Reading these next to the other branch's results

The merged `feature/reward-punishment` work (Pure Mycelia, autonomy sandbox, Symbiosis Engine, receptor memory) was produced in a separate session. **Its accuracy figures (for example 94.5% receptor memory and 93.7% Symbiosis Engine on the same brain-MRI data) have not been reproduced under the matched-readout protocol used here.** Treat them as claims until re-run. Its Pure Mycelia result (agents sharing verified discoveries solve unseen phrases at 300 evaluations where a single learner needs 20,000 or more) is the strongest "symbiotic" evidence in the repo, but it is measured on controlled symbolic worlds only.

Modules that can reach the network (`mycelia/reach/`, `mycelia/talk/llm.py`) are opt-in and should be reviewed before use; `reach` has no host allowlist by default.

Research use only; nothing here is a medical device or a clinical claim.
