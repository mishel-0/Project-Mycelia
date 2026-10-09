# MYCELIA adaptation audit and next mechanisms

## What the score means

The final 1.2.0 model processed all 7,200 original MRI images twice (14,400 lessons). Its 81.21% score is training/resubstitution recall, not its first-pass accuracy. A separate Training-only model achieved 76.89% on 1,467 clean Testing images. The 1.2.1 patch below does not retrain or modify either checkpoint.

The graph-memory experiment is online prototype learning in carbon- and energy-funded compartment state. It is not a validated fungal MRI-learning mechanism. The dimensional `mycelia.biology.Hypha` core remains separate and untrained. The memory lattice has its tips removed, so branching, fusion and changing sensing geometry do not currently contribute to MRI learning.

## Measured limitations

- **Compression:** only 32 colony imprints per class (128 total) represent 7,200 images. Lessons move the closest same-label imprint toward the new image instead of storing every image separately. Different appearances can become averaged together. Capacity alone does not determine whether perfect classification is possible.
- **Resolution:** images are reduced to 16x16 grayscale and gradients. Fine anatomy and original spatial detail are discarded. There were no conflicting labels for identical encoded inputs, so we cannot attribute the errors to known identical-input label contradictions.
- **Uneven use:** some imprints received only 2 exposures, while one received 440. Initialization admits the first 32 shuffled examples per class, without novelty-driven recruitment later. Some populations may therefore spend capacity poorly; this is a hypothesis to test, not an established cause of each error.
- **Plasticity slows:** consolidation decreases the receptor update cap to `1/(local_exposures+1)`. At 440 exposures the cap is about 0.00227, versus the initial subsequent rate 0.12. This favors retention but has no explicit mechanism to reopen plasticity after environmental change.
- **Cord saturation:** 18,848 of 61,440 cords (30.68%) reached their radius cap. A controlled changed-cue probe confirmed that the current rule then prevents contrast-trace updates, because trace updating is inside the positive-thickening-investment condition. This is a coupled modeling limitation, not a proven explanation of the MRI errors. A separate turnover mechanism must pay its own material/energy costs.
- **Weak demonstrated cord contribution:** removing learned cord traces retained 76.89% overall held-out accuracy. Some individual predictions changed, but there was no overall accuracy benefit. Receptor erasure caused all held-out predictions to abstain.
- **Energy snapshot:** no stored compartments had zero energy; the minimum final energy was about 0.985. There is no evidence from this final snapshot that simple energy exhaustion explains the score.
- **Teaching:** the correct label selects a population, but the rule does not detect that another label currently wins and recruit a distinct imprint in response. Labels and nutrient rewards remain an imposed experimental interface.

These observations identify constraints and hypotheses. They do not apportion the 18.79% training error causally. Perfect training recall could be obtained by a lookup table, but that would not establish biological adaptation or reliable unseen-image performance.

## Reproducible software bug fixed in 1.2.1

When two labels have identical imprints, reversing the saved `colonies` JSON dictionary made scalar prediction choose the second label while batch prediction chose the first. `scores()` now iterates canonical `labels` order. A regression test loads a reordered checkpoint, forces the tie, and checks scalar/batch agreement and unchanged frozen state. Training result version metadata now reads the package version instead of a hard-coded literal.

The recorded 7,200 recall rows contained zero exact score ties (minimum top-two gap about 3.22e-8). The tie bug is not an explanation for the reported 81.21% score. Separate patch verification replays the full recorded prediction sets and checks checkpoint hashes.

## Recommended next phase

1. **Fast and slow chemical state, with resource-paid turnover.** Keep a fast reversible local response alongside a slower persistent trace. Let sustained local stimulus change alter acquisition/turnover rates through explicit biochemical state, rather than an ever-growing exposure counter alone. Allow cord trace turnover even at maximum radius, with independent carbon/energy accounting. These equations are computational hypotheses requiring biological evidence and calibration.
2. **Adaptive capacity under a finite resource budget.** Recruit new sensing/memory structure when existing traces cannot represent a sufficiently different stimulus. Split/recycle underused structure conservatively instead of forcing every lesson into fixed capacity. Test this first as a hypothesis in the current graph module; call it prototype recruitment, not proof of fungal branching. New structures must inherit or consume accounted resources rather than receiving unexplained free biomass.
3. **A growing dimensional network.** Extend the new physiology core from its single apex to multiple locally regulated tips, then add contact-mediated fusion and conservative repair/pruning. Couple sensing, uptake, osmotic state, turgor, cargo supply, wall expansion and transport. Memory integration into this core is a separate implementation milestone; merely reusing old heuristic branching would not validate it.
4. **Transport that matters to function.** Make nutrient movement and source/sink demand control local reinforcement and recycling, and measure whether those changes improve resource delivery or adaptation. Current image-contrast reinforcement is not equivalent to physiological transport adaptation. Compare intact and disabled mechanisms at matched encoder, seed pools and resource budgets.
5. **Better image input as a separate engineering test.** Compare 16x16 with 32x32 and fixed multiscale sensing. This may preserve more useful detail; higher resolution itself is not a biological mechanism. Select all parameters using Training-only validation and keep resolution/resource effects separated.

## Tests before making claims

Run an A→B→A resource/stimulus switch. Measure the time to adapt to B, retention/recovery of A, material/energy expenditure, and the tradeoff between adaptation and forgetting. Repeat with memory, turnover, growth recruitment and transport reinforcement selectively disabled. Run injury/repair and resource relocation assays on grown networks. Then evaluate MRI recognition with held-out data, including a future patient-identified external dataset; the present folders cannot establish patient independence.

Success requires demonstrated adaptation and useful transport, not just a higher training score. Adding calcium, electrical channels or more named biological variables alone will not establish either. Each new state/law needs a target observable, a perturbation test and a species/strain scope.

## Biological evidence and boundaries

[Fukasawa, Savoury and Boddy (2020)](https://orca.cardiff.ac.uk/id/eprint/125667/104/s41396-019-0536-3.pdf) studied *Phanerochaete velutina* relocation and regrowth toward a previously encountered resource. This motivates directional-history assays, not a grayscale receptor learning law.

[Whiteside et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/31178314/) experimentally tracked phosphorus redistribution in arbuscular mycorrhizal fungi under unequal resource availability. This supports testing allocation/transport responses, but its species and symbiotic context cannot be assumed to calibrate a wood-decay fungus.

[Heaton et al., Growth-induced mass flows in fungal networks](https://arxiv.org/abs/1005.5305) proposes a model coupling growth, mass flow and adaptive transport. It motivates causal coupling and measured transport targets; it does not establish our MRI classifier or parameter values.
