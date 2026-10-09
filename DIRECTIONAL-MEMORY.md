# Phase 2: resource-gradient history and regrowth

## Candidate biological mechanism

Each compartment now has a persistent two-component polarity trace and a separate carbon-backed polarity-material pool. Actual uptake and the local environmental gradient determine acquisition; the mechanism never receives the true bait angle or a class label. It uses an explicitly hypothetical resource-associated directional acclimation rule.

A bounded uptake factor and gradient-strength factor set a rate `1-exp(-memory_rate * uptake_factor * gradient_factor * dt)`. The trace approaches the local gradient's unit vector. The requested polymer synthesis is `memory_cost * compartment_volume * vector_change_magnitude`, capped by local nutrient and ATP-like energy. Carbon transfers into polarity material and energy expenditure enters the existing memory ledger.

Exponential turnover decreases the trace and returns polymer carbon to reserve. Tip steering blends current polarity with the remembered direction in proportion to memory strength. Extension transfers a fraction of parent memory polymer to the new compartment and dilutes its trace; no extra polymer is created. Failed timesteps restore the complete state. State serialization includes these variables, with zero defaults for earlier saved states.

These equations are a candidate abstraction. They are not measured calcium, ion-channel, genetic or epigenetic mechanisms. Acquisition and directional coupling default to zero, so historical simulations remain behaviorally compatible.

## Experimental protocol

30 simulated trials; fixed configuration and seed 2026. Each trial has a randomized Gaussian bait direction. The growing network runs 120 steps in a field with baseline 0.2, amplitude 1.6, Gaussian sigma 8, and a center 12 grid units from inoculation. A uniform-history organism receives a field with the same spatial mean.

Only the root compartment is retained. All other tissue, old geometry, old tips and the training environment are removed from the probe. Each of six arms starts with identical physical pools from the trained root, with only the directional state or its coupling varied. This resource matching is an artificial causal control. The sham substitutes the directional trace from the uniform-history organism.

The probe is a fresh uniform substrate with nutrient 0.6 and water 8. Four fresh tips have a randomly rotated polarity independent of the old bait angle, paired across all arms. New memory acquisition is disabled during the 100-step probe. Existing memory may decay and be inherited.

Arms: intact memory; erased trace; steering gain zero; inheritance zero; trace rotated 180 degrees; uniform-history sham. The primary statistic is the paired difference in length-weighted cosine alignment of new segments with the former bait direction, intact minus erased. A 10,000-resample paired bootstrap describes uncertainty over simulated trials.

## First outcome and diagnosis

The primary confidence interval includes zero: **this candidate did not demonstrate reliable former-resource directional memory**. Its acquisition rate, costs and gain were not retuned after observing that result. Instrumentation added afterward reproduced the same numerical outcome and exposed the acquisition history.

The stored trace initially aligns with the resource direction, then drifts as local substrate gradients change. A simple latest-gradient moving trace is insufficient to establish stable memory of the original resource. The arm with steering disabled matches the erased arm, confirming where the coded bias enters, without proving biological validity.

No training trial reached within 3 grid units of the bait center. Furthermore, the Gaussian field supplies some resource at the original inoculum from the outset. Thus this is a **gradient-history regrowth assay**, not a reproduction of colonization of a discrete wood bait followed by transplanting a wood inoculum. The single retained compartment also omits distributed tissue architecture that may matter biologically.

The benchmark is motivated by [Fukasawa et al., ecological memory in P. velutina](https://pubmed.ncbi.nlm.nih.gov/31628441/), which investigated resource colonization and subsequent regrowth. Our root-only, resource-matched computer assay is deliberately different and cannot establish empirical agreement with that study.

## Next modeling requirements

1. Require and record colonization of a discrete finite resource before the transplantation phase; distinguish trials that fail to reach it.
2. Preserve a spatial inoculum region and test architecture separately from biochemical state, rather than assuming one compartment captures ecological memory.
3. Compare instantaneous local-gradient traces with successful-resource/transport-history consolidation hypotheses; test new rules on new directions/seeds and retain negative results.
4. Fit growth rates, tip/cord geometry, resource depletion and transport to species-specific measurements. A statistically positive synthetic result would still validate only the implemented candidate behavior.

No CNN, neural network, SVM or MRI classification is involved in this experiment. All constants use uncalibrated model units.
