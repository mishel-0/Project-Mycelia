# Phase 2: candidate adaptive graph memory

## Scope

This milestone adds resource-paid fast/slow chemical state and independent cord-trace turnover to `AssociativeMycelium` and `MemoryColony`. It is an **opt-in, uncalibrated computational hypothesis** on the existing prototype lattice. The dimensional `mycelia.biology.Hypha` core remains separate and untrained. Branch recruitment, growing MRI sensing geometry, fusion-driven learning, explicit ions/calcium, electrical memory and empirical species calibration are not implemented in this milestone.

The earlier 81.21% full-data training recall and 76.89% unseen-image accuracy belong to the preserved legacy model. They must not be presented as scores of this new mechanism.

## Debugging gate completed first

The existing 109 tests passed. Additional probes exposed acceptance of NaN resource budgets, duplicate checkpoint IDs, malformed freeze flags and obstacle masks, plus learning/recognition by retired tissue. These were fixed before adding adaptive state. Ledger values, graph allocation identities, masks, life flags and physical geometry now receive stronger validation. Inactive compartments/cords cannot acquire memory; recall ignores retired compartments and abstains when no living funded imprint remains. Loaded scalar fields cannot silently override other organism objects.

The debugging gate then passed 142 tests, 540 randomized memory operations across 18 configurations, and 180 dimensional physiology steps across 12 resource/pressure conditions. The added mechanism passed a further 17 regression cases. Legacy checkpoint replay checks that the original held-out and all-data predictions remain unchanged, without modifying their saved files. These checks verify the tested invariants, not an absence of every possible bug.

## State and equations

Every compartment now owns a slow `receptor_trace` and a fast `receptor_fast_trace`, each with its own finite carbon-backed material and acquisition count. Existing checkpoints load with empty fast pools. Every cord has slow/fast contrast traces with separate marker-material pools, distinct from structural thickening material and radius.

Let `x` be the local three-channel cue, `f` the previous fast trace and `s` the slow trace. Defaults are fast acquisition `a_f=0.45`, slow acquisition `a_s=0.015`, and stability scale `k=0.02`. For previously funded state:

```text
g = exp(-mean((x-f)^2) / k)
a_fast = a_f * reward
a_slow = a_s * g * reward
request = material_cost * a * (0.05 + mean(abs(x-trace)))
paid = min(request, local_nutrient, local_energy / energy_cost)
a_actual = a * paid / request
trace += a_actual * (x-trace)
```

The first funded imprint uses rate 1, scaled by reward and affordability. The stability gate is 1 when no fast state exists; a zero acquisition rate disables that channel completely. Setting fast or slow acquisition to zero therefore supplies genuine single-channel controls. Consolidation counters do not progressively shut off adaptive-mode acquisition. The equations are local, with no backpropagation or external learned readout.

Each paid node update transfers nutrient into that channel's material pool, spends `paid * energy_cost` and credits the shared energy-spending ledger. Fast and slow channels are charged separately. Slow acquired variation remains a stored statistic; it is not a diagnostic likelihood.

Cord marker updates use adjacent-cue contrast and the same fast/slow acquisition idea, with `request = route_memory_cost * a * (0.05 + abs(contrast-trace))`. They draw equal carbon and energy shares from both adjoining compartments. Marker updates can continue when the physical cord radius is at its cap. Structural thickening remains a separate paid update. Neither channel can change without affordable carbon and energy.

`rest(duration)` decays fast traces/material at 0.03 per prototype model-time unit and slow traces/material at 0.003. Recycled carbon returns to reserve; energy already spent is not refunded. Cord marker pools follow the corresponding decay rates; structural material follows the existing slow recycling rule. Retraction returns all added marker material conservatively. These are prototype units, not measured physical reaction rates.

Recognition takes the minimum direct graph mismatch over the available, living, funded channels within each colony, then across colonies for each label. Each channel is a separate stored imprint. A dual-channel model therefore has more state/templates than a single-channel model at identical colony count. Batch arrays are rebuilt from graph state and have no separately trained predictor cache. Frozen inference is read-only.

## Reproduce the adaptation assay

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python tools/adaptation_test.py --output runs/adaptation --trials 30 --size 8
```

The seeded paired assay supplies 40 identical A cue exposures, 12 B exposures, then 12 A exposures. Five arms share initial geometry, resource pools, feed and physiology schedule: fixed legacy acquisition, aging legacy consolidation, dual fast/slow, fast-only and slow-only. Material expenditure can differ and is reported. All groups see exactly the same cue pair within each trial.

Distances are normalized by the common per-trial A–B cue mean-square difference. B adaptation requires mismatch <=0.05; A retention, measured **before reteaching**, requires mismatch <=0.10. These are operational computational thresholds, not clinical accuracies. B latency is censored after 12 exposures; failures are recorded as missing and coded as 13 only for the explicitly restricted paired comparison. The A return phase measures recovery; zero latency means A was already retained before re-exposure. Full trajectories and per-trial values are saved. The scaffold does not grow during this assay, and nutrient feed remains matched; this is a chemical-cue switch, not a wood-bait or physical nutrient-foraging experiment.

Thirty trials passed both adaptation/retention checks for the dual model. Its B threshold was reached in three exposures in each trial. Fast-only also adapted in three exposures but failed A retention. Fixed acquisition adapted in 27/30 trials within 12 exposures and failed the A-retention threshold; aging and slow-only arms did not adapt to B within the horizon. The dual model spent more synthesis material, about 7.06 model units per trial versus 6.29 for fixed acquisition. Maximum budget error remained about 1e-12. Paid-radius saturation fixtures confirmed marker updates at unchanged physical radius.

The assay supports the behavior of these implemented equations on these stimuli. It does not establish their fungal molecular mechanism or MRI usefulness, and returning to A can overwrite recently acquired B in the fast channel.

## Separate MRI validation

```bash
python -m pip install -e '.[images]'
python tools/adaptive_medical_validation.py --dataset Brain-Tumor-MRI-Dataset --output runs/adaptive-mri-validation
python tools/predict_memory_image.py --memory runs/adaptive-mri-validation/adaptive-validation-memory.json --image /path/to/image.jpg
```

The exploratory comparison uses 2,000 inner-Training images for two passes and 1,082 validation images drawn from clean Training. It does **not** score the independent Testing folder or fit a new final all-data model. Candidate rates are specified before MRI validation. Same colony count does not equal the same count of stored templates; dual state carries an explicit capacity/cost advantage that must be separated in future controls. Exact source and encoded-input duplicate filtering remains; patient independence and clinical validation are absent. See its `summary.json` for the actual result; no accuracy increase is guaranteed by better adaptation.

The completed comparison scored **78.19%** for the previous aging rule, **76.62%** for the dual candidate and **69.32%** for fast-only acquisition. Balanced accuracies were 78.29%, 76.79% and 69.46%, respectively. The previous rule's validation result reproduced exactly. Improved chemical-cue adaptation did not improve this MRI validation score; the previous trained models remain preserved and the candidate remains opt-in.

The default memory configuration retains the legacy rule. The experimental switch is:

```python
from mycelia import MemoryConfig, AssociativeMycelium
config = MemoryConfig(adaptive=True, colonies_per_label=32)
model = AssociativeMycelium(['glioma', 'meningioma', 'notumor', 'pituitary'], config)
```

## Biological rationale and remaining gates

[Directional history in Phanerochaete velutina](https://orca.cardiff.ac.uk/id/eprint/125667/104/s41396-019-0536-3.pdf) motivates studying persistent effects of previous resource exposure. [Drought priming in two filamentous fungi](https://pubmed.ncbi.nlm.nih.gov/31950228/) motivates stimulus-history and species-specific adaptation assays. Neither paper establishes these two-trace equations, grayscale-to-chemical encoding, human label rewards or their rates.

The next integration gate is a growing dimensional network with conserved branching and contact-mediated fusion, followed by transport-dependent reinforcement/repair and memory coupling. Actual species/strain observables and calibrated parameters are needed before claims of biological resemblance beyond the stated abstraction.
