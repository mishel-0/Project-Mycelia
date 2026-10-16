# Phase 9 — Electrical-oscillation-style global signaling (SELFO analogy)

**Setup**: trained on 5 books (2.35M chars), tested on Moby Dick held out **entirely** — genuine domain shift, not a random split. Online/prequential evaluation (predict-then-learn, char by char) so the "reacts faster to novelty" mechanism actually had the chance to do something — the first version of this test used frozen scoring and gave it zero opportunity, which was itself a bug worth catching.

**Mechanism tested**: a network-wide `stress` signal (bounded EMA of recent surprise, [0,1]) that decays over time and temporarily lowers the specialist-recruitment threshold everywhere when the network has recently been confused — modeling how a real SELFO spike lets distant parts of a fungal network react without their own local evidence yet.

## Result: GATE FAILED — stress signal had literally zero measurable effect

| | Baseline | Plain grower | Stress-signaled |
|---|---:|---:|---:|
| Mean bits/char (whole held-out book) | 2.8408 | 2.8415 | **2.8415 (identical to plain)** |
| Mean bits/char (first 5,000 chars, hardest novelty) | 4.0278 | 4.0330 | **4.0330 (identical to plain)** |

`stress_signaled` vs `plain_grower`: delta = **0.0 exactly**, CI = [0, 0]. The mechanism changed nothing.

## Why — a structural finding, not a bug
Verified `stress.stress_history` does spike correctly on style shifts (confirmed in a standalone sanity check before this run). The problem is deeper: **the recruitment gate is already saturated for genuinely novel text.** When a context has never been seen, the fallback prediction is near-uniform over the alphabet (~76 characters), so surprise on the actual character is already ≈0.99 — far above the base `novelty_threshold` (0.6) *before* any stress-based reduction is applied. Lowering the threshold from 0.6 toward the floor (0.1) changes nothing, because 0.99 clears either value with room to spare. The network was already recruiting a specialist for essentially every new 4-gram it met, with or without stress.

**Implication**: modulating the recruit/no-recruit threshold is the wrong lever for "react faster to novelty" in this architecture — it already reacts maximally. A mechanism that could plausibly matter instead: stress modulating the *learning rate or forgetting rate* of existing specialists (so recently-stressed regions adapt their counts faster / discount older evidence), or modulating how much weight a brand-new, low-count specialist gets versus the global fallback when predicting (currently a specialist with even 1 observation fully overrides the fallback). Neither of those has been tested yet.

## Running tally across every mechanism tested so far
Structural growth (Phase 8), continual learning (Phase 2), few-label learning (Phase 3), and now global stress signaling (Phase 9) have all failed to beat a trivial baseline, each for a diagnosable structural reason rather than random noise. Phase 4 (masked image-patch prediction) remains the only validated MYCELIA-specific win.
