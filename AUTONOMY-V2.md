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
