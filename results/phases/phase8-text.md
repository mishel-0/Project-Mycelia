# Phase 8 — Text domain, growing character-colony

**Data**: 6 named public-domain books streamed from Project Gutenberg (Alice in Wonderland, Frankenstein, Pride and Prejudice, Moby Dick, Huckleberry Finn, The Picture of Dorian Gray) — 3.58MB combined, cached locally in `runs/phase8-text/corpus/`. Each source is a specific, named file, not an open-ended crawl.

**Gate**: MYCELIA's growing character-colony (`mycelia/text_memory.py`) beats a trivial fixed-order n-gram baseline on held-out text (90/10 split), measured in bits/character, with 95% paired bootstrap CI excluding zero.

## Result: GATE FAILED — and found something more informative than a simple loss.

| Run | Corpus | Capacity | MYCELIA bits/char | Baseline bits/char | Delta (95% CI) |
|---|---|---:|---:|---:|---|
| Underfunded | Alice (130K train chars) | 4,000 (< 19,133 real contexts) | 4.487 | 3.446 | **+1.04** [1.018, 1.063] — clearly worse |
| Properly sized | Alice (130K train chars) | 20,000 (train-only context count) | 3.44634 | 3.44634 | **−0.0000026** [−7.8e-6, 0] — statistically identical |
| Properly sized, full scale | 6 books (3.16M train chars) | 100,000 | 2.378664 | 2.378664 | **−7.3e-8** [−2.2e-7, 0] — statistically identical |

## What this means
1. **Bug found and fixed along the way**: the original implementation re-sliced the entire growing text prefix on every character (`text[:i]`), making it O(n²) — would never have finished on 3.6MB. Fixed to a rolling fixed-width context window.
2. **Underfunded capacity is actively harmful**: when `max_specialists` is smaller than the number of real distinct contexts, the colony thrashes (42,904 recruited vs. 38,904 recycled on just 130K characters) and ends up *worse* than the dumb baseline.
3. **Properly funded, it converges to being identical to the baseline — not better.** At both scales tested (130K chars, 3.16M chars), once capacity matches the real context count, MYCELIA's growth/recruitment mechanism reconstructs the exact same lookup table a trivial fixed n-gram model builds directly, just via a more complicated resource-gated process. The novelty-threshold recruitment rule effectively ends up creating a specialist for almost every context anyway, so there's no compression or generalization benefit over the straightforward table.

## Honest conclusion
The "self-growing architecture" hypothesis, as implemented, does not produce a text model that's better than a baseline you could write in 15 lines without any growth/recycling logic. It doesn't hurt once properly sized, but it also doesn't help. This mirrors the pattern from Phases 1–3: MYCELIA's biological machinery (growth, recruitment, pruning) has not yet been shown to add predictive value anywhere it's been tested — image classification, continual learning, few-label learning, or now text prediction. Phase 4 (masked image-patch prediction) remains the only capability where MYCELIA measurably beat a simple baseline.

**What would actually test the hypothesis fairly**: a scenario where the *fixed* baseline cannot be built directly — e.g., unbounded/streaming vocabulary where you don't know the context count in advance and must cap memory below the true table size while losing as little accuracy as possible. That's a real resource/accuracy tradeoff question growth+recycling could in principle help with, and it has not yet been tested here.
