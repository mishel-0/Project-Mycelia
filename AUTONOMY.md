# Mycelia Autonomous Sandbox v0.1 — symbolic driving

First test environment for Mycelia as a coordination layer for autonomous
systems. No neural networks, pretrained models or AI APIs. A toy simulator
for studying knowledge transfer; it says nothing about real vehicle safety.

Code: `mycelia/autonomy/` (`sim`, `scenarios`, `node`, `network`),
`tools/drive_experiments.py`, `tests/test_autonomy.py`. Eval suites were
frozen with a SHA-256 manifest and committed (bc4aa02) before the learners
were written. Reproduce: `PYTHONPATH=.:tools python tools/drive_experiments.py`.

## Setup

- **Simulator:** 24-cell road, 1–3 lanes; static blocks, crossing
  pedestrians, slow vehicles; clear (6-cell view) or fog (2 cells, side lanes
  unknown); dry or wet (a stop while moving rolls one more cell). Cars see
  only symbolic features: what is ahead, how near, whether side lanes are
  free/busy/unknown/absent, visibility, surface. Built-in reflex: stop when
  something is near.
- **Cars (4):** each practises in its own family — city blocks, rain with
  pedestrians, fog traffic, highway traffic — with 300 episodes.
- **Learning:** when an episode fails or wastes time, the car experiments
  with each action for that situation on 6 variations against the reflex,
  keeps an action only if it never collides and is better, then generalises
  by dropping conditions one at a time while it still holds. Dropped
  conditions keep a **scope**: the values actually seen while testing.
  Examples learned: "block ahead + right lane free → move right" (scope:
  clear, dry); "fog + vehicle ahead → slow" (scope: dry); overtaking rules
  on the highway (scope: clear, dry).
- **Evaluation:** 3 frozen suites × 400 scenarios (4 practice families +
  4 held-out: one-lane road with a block, fog with blocks, rain with
  traffic, random mixes). Every car drives every scenario: 4,800 episodes
  per condition.

## Results

| Condition | Success | Collisions | Collisions involving another car's rule | Rules from others that led to success |
|---|---|---|---|---|
| A isolated | 58.9% | 92 | 0 | 0 |
| B shared repository | **71.4%** | 368 | 276 | 1,641 |
| C fixed sharing (ring) | 67.3% | 276 | 184 | 1,096 |
| D adaptive symbiosis | 66.6% | **12** | 9 | 1,048 |
| A_scope (isolated + scope check) | 57.7% | **3** | 0 | 0 |
| B_scope (repository + scope check) | 66.6% | **12** | 9 | 1,054 |

Held-out families (success / collisions):

| | A | B | C | D |
|---|---|---|---|---|
| One-lane road with block (only safe outcome: stop) | 0% / 50 | 0% / **200** | 0% / 150 | 0% / **0** |
| Fog + block | 4.5% / 23 | 18% / 92 | 13.5% / 69 | 0% / **0** |
| Rain + traffic | 100% / 0 | 100% / 0 | 100% / 0 | 100% / 0 |
| Mixed | 42.5% / 16 | 55.3% / 64 | 51.2% / 48 | 43.3% / **0** |

## What this shows (and doesn't)

1. **Sharing knowledge helps.** Cars driving other cars' families: 75%
   success alone vs 99.5% with a shared repository.
2. **Blind sharing is dangerous.** Over-generalised rules ("block ahead →
   change lane") were applied where they can't work, such as a one-lane road
   (200 crashes) and fog (92).
3. **Knowing when not to apply a rule removes almost all crashes** (368 → 12)
   at a cost in success (71.4% → 66.6%). In fog and one-lane cases the car
   now waits safely instead of crashing.
4. **The adaptive part of D added nothing measurable.** The ablation B_scope
   (shared repository + scope check, no re-verification, no trust learning)
   produced identical numbers to D. Re-verification passed all 11 rules it
   tested and never rejected one, so trust learning never triggered. **The
   benefit comes from verified scope, not adaptive symbiosis.**
5. D and the scope-checked conditions lose some own-family success (99.5% →
   97.3%) because strict scope refuses some valid uses; D also practises
   with 60 fewer episodes (reserved for re-verification).

## Next experiments that would test adaptive symbiosis properly

Scenarios where a sender's rule is *in scope* but still wrong for a
receiver (hidden differences such as braking dynamics), unreliable or lying
cars, many more cars than rules can be checked for, and a cost per message.
Adaptive trust and re-verification can only matter when scope checks alone
cannot catch the problem.
