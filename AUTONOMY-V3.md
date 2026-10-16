# Mycelia Autonomous Sandbox v0.3 — biologically inspired mechanisms (pre-registration)

Committed with the frozen suites (`datasets/drive_v3_eval_{1,2,3}.json`,
manifest `datasets/drive_v3_manifest.json`) before any v0.3 mechanism was
written. Results will be appended below without editing this section.
Same world as v0.2 (hidden brake differences, a misleading car, hidden shift
after round 2, counted messages). Toy simulator; no neural networks,
pretrained models or AI APIs; no claims about real vehicles.

## Mechanisms (each switchable; biology is inspiration, not a claim)
- **Rat (world model):** per-car table of (situation, action) → outcomes,
  updated after every drive; used to choose actions when no rule applies;
  failure analysis also experiments at earlier moments before the failure.
- **Crow (procedures):** rules may generalise across obstacle types when that
  still holds in tests.
- **Octopus (local veto):** a car refuses any rule whose action its own
  experience shows crashing in that situation.
- **Honey bee (probation + quorum):** shared rules are advertised by evidence,
  used on probation, trusted network-wide after 2 independent confirmations,
  withdrawn after a crash.
- **Slime mold (reinforce/prune):** rules and car links carry a strength that
  grows with useful outcomes, decays and is pruned; message capacity follows
  link strength.

## Conditions
A isolated (v0.1 learner) · M3 (best v0.2 condition) · BIO_full (all five) ·
BIO_full minus each mechanism (5 ablations). Equal episode budgets; 3 seeds ×
3 frozen suites.

## Primary criterion
BIO_full "helps" only if it beats **M3** (the strongest previous system) on
success or collisions with non-overlapping Wilson 95% intervals, without the
other metric worse beyond its interval. A mechanism "contributes" only if
removing it makes BIO_full worse by non-overlapping intervals.

## Secondary measures
Success and collisions per obstacle type (block, pedestrian, vehicle) and
per car; episodes needed until a car first handles each obstacle type in
operation; transfer to other families and held-out families; operational
collisions per round around the shift; messages; false rule acceptance.
