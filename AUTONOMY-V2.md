# Mycelia Autonomous Sandbox v0.2 — pre-registration

Written and committed together with the frozen evaluation suites
(`datasets/drive_v2_eval_{1,2,3}.json`, SHA-256 in
`datasets/drive_v2_manifest.json`) **before** any v0.2 mechanism was
implemented. Results will be appended below this section without editing it.

## Question
Can adaptive coordination learn which knowledge to trust, when to share it
and when to reject it — even when a rule looks applicable — better than the
best non-adaptive system from v0.1, `B_scope` (shared repository + verified
scope checks)?

## Challenges (hidden from observations)
1. Capabilities: car1 strong brakes (city blocks), car2 weak brakes (rain,
   pedestrians), car3 normal (fog traffic), car4 strong (urban, pedestrians).
2. Misleading node: car5 shares corrupted rules with inflated scope and
   fabricated evidence.
3. Hidden shift after round 2: worn tyres (+1 cell rolled after stopping) and
   pedestrians linger 3 ticks longer in city_blocks, rain_pedestrians and
   urban_mixed.
4. Communication cost: messages and bytes counted; M4 limits each car to k
   rule messages per round.

## Conditions
A isolated · B repository · B_scope · B_scope+M1 (re-verification on the
receiver's own scenarios and brakes) · B_scope+M2 (source trust) ·
B_scope+M3 (runtime monitoring and retraction) · B_scope+M4 (value-based,
bandwidth-limited sharing) · D_full (M1–M4). Same total episode budget per
car (verification spends from it). 4 rounds; 3 seeds × 3 frozen suites.

## Primary criterion
"Adaptive coordination helps" only if D_full beats B_scope on collisions or
success with non-overlapping Wilson 95% intervals over all evaluation
episodes, and the other metric is not worse beyond its interval. Otherwise
the result is reported as null.

## Secondary measures
False rule acceptance (adopted foreign rules that an oracle check on the
receiver's own post-shift scenarios and brakes shows unsafe), unsafe
transfers (collisions where a foreign rule fired), unnecessary waiting
(idle ticks), recovery after the shift (operational collisions per round),
messages and bytes, success on held-out families. Every single mechanism is
reported, including those that do not help.

---

# Results (appended after the run; the section above is unchanged)

Code: `mycelia/autonomy/{trust,network_v2}.py`, `tools/drive_experiments_v2.py`,
`tests/test_autonomy_v2.py`; raw results `results/autonomy-v2/drive-v2-results.json`.
Reproduce: `PYTHONPATH=.:tools python tools/drive_experiments_v2.py`.
4 cars × 320 scenarios × 3 seeds/suites = 3,840 evaluation episodes per condition.

| Condition | Success (95% CI) | Collisions (rate CI) | Unsafe transfers | Liar rule fired | Foreign rules unsafe for holder (oracle) | Messages |
|---|---|---|---|---|---|---|
| A isolated | 33.6% (32.1–35.1) | 1,741 (43.8–46.9%) | 0 | 0 | 0 | 0 |
| B repository | 35.8% (34.3–37.4) | 2,464 (62.6–65.7%) | 2,203 | 2,546 | 11 | 78 |
| **B_scope** | 33.9% (32.4–35.4) | 2,538 (64.6–67.6%) | 2,415 | 2,757 | 11 | 78 |
| +M1 re-verification | 33.5% (32.0–35.0) | 1,701 (42.7–45.9%) | 195 | 431 | 1 | 120 |
| +M2 trust | 33.9% | 2,538 | 2,415 | 2,757 | 11 | 78 |
| **+M3 monitoring** | **46.8% (45.2–48.4)** | **1,315 (32.8–35.8%)** | 307 | 775 | 2 | 75 |
| +M4 value sharing | 33.9% | 2,538 | 2,415 | 2,757 | 11 | 78 |
| D_full (M1–M4) | 33.4% (31.9–34.9) | 1,721 (43.3–46.4%) | 341 | 651 | 1 | 90 |

Operational collisions per round (sum over seeds; shift before round 3):
A 36/32/40/43 · B_scope 104/89/101/109 · M1 38/32/46/42 · M3 39/32/40/43 · D_full 37/33/47/42.

## Primary criterion (as pre-registered)

**Met:** D_full has fewer collisions than B_scope with non-overlapping
intervals (43.3–46.4% vs 64.6–67.6%) and success is not worse (overlapping).

## What it actually means

1. **The win is defence against the misleading car.** B_scope adopts the
   liar's rules (they claim universal scope), so scope checking alone gives
   no protection: 2,757 episodes in which a liar rule fired. M1
   re-verification on the receiver's own world cut that to 431 and false
   acceptances from 11 to 1.
2. **D_full is not better than isolated cars** (1,721 vs 1,741 collisions,
   33.4% vs 33.6% success). In this setup the network's net benefit over not
   sharing at all is roughly zero: adaptive mechanisms mainly cancel the harm
   that sharing introduced.
3. **M3 runtime monitoring alone is the best condition by a wide margin**
   (46.8% success, 1,315 collisions), better than D_full and better than
   isolated cars. Retracting rules after a collision removes the liar's rules
   and own or foreign rules that the hidden shift made unsafe.
4. **Combining mechanisms made things worse than M3 alone.** In D_full the
   verification reserve takes practice budget, verification rejects or
   ignores rules before monitoring could judge them in use, and fewer
   retractions happen (4 vs 7). Mechanisms interact; more is not better.
5. **M2 and M4 alone changed nothing.** Trust only gates future adoption and
   every liar rule was adopted in round 1; M4's ranking trusted the liar's
   fabricated evidence (50/50 tests), so bandwidth limits selected the liar.
6. Overall collision rates are high (≥33%) because the reflex itself is
   unsafe for weak brakes on wet or shifted roads; no condition learned a
   general fix for that in 400 episodes.
7. Development note: a first smoke run showed M1 never verifying because
   practice consumed the whole budget, and M1 accepting liar rules that never
   fired during verification; both were fixed before the evaluation run
   above (reserve budget; a rule must fire at least once to count as verified).

**Answer to the v0.2 question:** experience-based coordination helps in one
specific way — **monitoring rules in use and retracting them after failures**
handles misleading knowledge and hidden change better than any static
scheme here. Source trust and value-based sharing, as implemented, did not
help. The network as a whole still does not beat isolated learners except
through M3.
