# Mycelial Symbiosis Engine

An experimental learner with no fixed architecture: the set of memory traces
*is* the model, and it grows, strengthens, weakens, merges and prunes itself
while learning, under a metabolic budget, and can heal after damage.

Code: `mycelia/symbiosis.py`, experiment `tools/symbiosis_mri.py`,
tests `tests/test_symbiosis.py`, results `results/symbiosis/symbiosis-results.json`.

> Computational hypothesis inspired by fungal networks. Not consciousness, not
> clinical software. Every claim below is a measured number on one dataset.

## Mechanisms implemented

| Mechanism | What it does here |
|---|---|
| Structural plasticity | A new trace grows only when recall is wrong or unsure (margin < threshold). Traces that keep misleading are pruned. |
| Local reward/punishment | Only the top contributing traces of each decision get credit: helpful ones strengthen, misleading ones weaken (floor, not deletion, so they can recover). |
| Metabolic budget | At most `budget` traces; every recall costs one energy unit per trace consulted; the least useful fast trace is evicted first. |
| Fast/slow memory + sleep | New traces are fast. Every 1,000 steps, sleep merges proven fast traces into a similar slow trace of the same label (or promotes them), prunes misleading ones, and decays utility. |
| Self-healing | Compartments or traces can be destroyed. Healing re-expresses memories through the surviving compartments and replays experience once, so failures regrow traces. |

Not yet implemented from the proposal: stigmergy, specialist evolution,
gossip exchange, immune system, curiosity, collective dissent.

## Setup

Input is the 6×6 compartment response from `tools/mycelial_network_mri.py`;
each compartment compresses its own region to 16 numbers (unsupervised
PCA, 576 per scan). Settings (gamma, growth margin, epochs) were chosen on a
20% validation split of Training (selected: gamma 30, margin 0.5, 2 passes;
validation 96.2%). Testing: 1,467 held-out images. Runtime 7.5 minutes.

The comparison "fixed network" stores all 10,820 training memories (images
plus mirrors) with weights from one kernel-ridge solve and never changes.

## Headline (held-out Testing)

| | Engine | Fixed network |
|---|---|---|
| Diagnosis accuracy | 93.7% | **94.6%** |
| Tumor presence accuracy | **97.4%** | 97.1% |
| Tumor sensitivity / specificity | 96.5% / 100% | 96.1% / 100% |
| Memories stored | **1,805** | 10,820 |
| Recall cost per image | **1,805** | 10,820 |

The engine is 0.9 points lower on 4-class diagnosis with **6× less memory and
recall compute**, and slightly better at tumor presence.

## Ablations (Testing)

| Variant | Accuracy | Traces |
|---|---|---|
| Full engine | 93.7% | 1,805 |
| No plasticity (store everything until budget) | 92.4% | 4,000 |
| No local credit | 93.6% | 2,404 |
| No sleep | 93.5% | 2,210 |

Plasticity is the mechanism that matters (+1.3 points with fewer traces).
Local credit and sleep mainly shrink memory (−25% and −18% traces) with
about the same accuracy; their accuracy effect is within noise.

## Experiment C: compute budget

| Max traces | Accuracy |
|---|---|
| 100 | 74.4% |
| 250 | 80.5% |
| 500 | 82.8% |
| 1,000 | 87.7% |
| 2,000 (uses 1,805) | 93.7% |

## Experiment A: damage and self-healing (mean of 3 random damages)

Compartment death (signal from those image regions is lost everywhere):

| Killed | Fixed network | Engine damaged | Engine after healing |
|---|---|---|---|
| 10% | **94.5%** | 93.2% | 93.7% |
| 30% | **93.9%** | 92.4% | 93.4% |
| 50% | 91.6% | 89.0% | **92.7%** |

Memory-trace loss:

| Lost | Fixed network | Engine damaged | Engine after healing |
|---|---|---|---|
| 10% | 45.3% | 92.3% | **93.5%** |
| 30% | 35.1% | 89.2% | **93.5%** |
| 50% | 30.3% | 85.0% | **93.5%** |

Findings:
- After losing half its compartments, the healed engine (92.7%) beats the
  fixed network (91.6%), recovering from 89.0%.
- After losing memories, the engine returns to full accuracy (93.5%) at
  every damage level. The fixed network collapses because its kernel-ridge
  weights depend on each other and are never re-solved.
- Caveats: healing replays the training set once (data must be available);
  the fixed network could also be re-solved after damage, which would restore
  it at the cost of a full retrain. Its collapse shows brittleness of fixed
  interdependent weights, not that it cannot be retrained.
