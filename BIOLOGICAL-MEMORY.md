# Biological memory model and evidence

## Evidence versus hypothesis

Experimental work reports ecological resource memory and directional regrowth in *Phanerochaete velutina*: [Fukasawa et al., ecological memory (2020)](https://pubmed.ncbi.nlm.nih.gov/31628441/). Research also reports resource-efficient network changes through selective reinforcement and recycling: [Bebber et al., biological transport-network design (2007)](https://pubmed.ncbi.nlm.nih.gov/17623638/).

These studies support investigating resource-dependent persistent adaptation. They do not establish that fungi encode grayscale MRI pixels, recognize tumors, use the receptor equations below, or learn human class labels. All those interfaces and equations are our computational hypotheses. This release is a mechanistic research prototype, not a biologically validated replica.

## Where the learning lives

Each real `Compartment` stores a three-channel receptor trace, encountered variation, exposure count and a carbon-backed memory-material pool. Each real `Segment` stores a local contrast trace, physical reinforcement material, radius and septal opening. These values persist through the organism's JSON state and are included in its conservation checks. There is no learned external classifier in this experiment. Class names are assigned to rewarded colony populations by the experimenter.

The stimulus interface converts fixed-scale grayscale intensity and signed spatial gradients to three nonnegative chemical cues. It is a deterministic encoder with no learned parameters. Training initializes up to the configured number of local imprints per label and subsequently rewards the closest existing colony within that label. The original `network_memory_test.py` uses eight imprints per label and one shuffled exposure pass. The newer `train_mycelium.py` chooses among three configurations on a Training-only validation split, tests a separate Training-only model, and then fits a final all-data model; see `FULL-DATA-TRAINING.md`.

For local cue `s`, trace `m`, rate `a`, and reward `r`, requested trace synthesis is `c = material_cost * a*r * (0.05 + mean(abs(s-m)))`. Actual synthesis is capped by local carbon and available energy. The rate scales down if resources are insufficient. The trace updates as `m += affordable_rate * (s-m)`; encountered variation follows an analogous local moving statistic. The first affordable imprint can adopt the cue; subsequent ones use rate 0.12 by default. Optional consolidation caps that rate at `1/(previous_local_exposures+1)`, making frequently exposed compartments adapt more slowly. This is a computational consolidation hypothesis, mathematically an online prototype update, not a discovered fungal learning law. The variance statistic is stored but is not used as a diagnostic likelihood.

Synthesis transfers local nutrient to memory material and spends ATP-like energy in a dedicated ledger. Cord consolidation transfers nutrient into the existing segment-material pool, spends energy from both adjoining compartments, changes actual radius and therefore hydraulic conductance, and updates a local contrast trace. Version 1.2.0 corrects the previously omitted cord energy cost. Every twentieth exposure executes the full physiological timestep, including transport, damage regulation and remodeling. The scaffold has no active tips; its lattice geometry is seeded, not grown during this classification experiment.

Recognition is the minimum across each label's rewarded colonies of `mean((incoming_cue - receptor_trace)^2) + 0.15 * conductance_weighted_cord_contrast_mismatch`. This is an explicit direct-match decision rule. Numerical mismatch scores are not diagnostic probabilities. No backpropagation, dense neural layers, CNN or SVM is involved.

`rest(duration)` decays chemical traces toward background and recycles memory/cord material to reserve while conserving carbon. Recall is read-only; freezing prevents teaching or turnover. Failed exposure and turnover roll back resources, environment and physiological state. A new colony is registered only after a successful paid lesson. An unfunded or erased imprint abstains. Batch recall reconstructs its temporary arrays from actual node and cord state on every call, so there is no separately fitted or stale cached predictor.

## What this release tests

The suite checks internal imprints, actual cord thickening, resource costs, lack of learning without resources/reward, turnover and forgetting, conserved carbon/water/energy, read-only frozen recall, JSON save/resume and failed-exposure rollback. The MRI run compares held-out recognition with a majority baseline, removal of receptor memory, and removal of learned cord traces.

The model stores useful imprints; this does not show that every proposed biological mechanism causes recognition. Ablations must be reported, including small or absent contributions from reinforced cords. The new training workflow removes exact source-image and actual 16x16 encoded-input collisions; patient-independent validation and transformed-duplicate removal are not established. The dimensional `mycelia.biology.Hypha` core is separate and is not trained by this graph-memory workflow.

The legacy physiology remains an uncalibrated model. Molecular ion channels, calcium signaling, gene expression, actual epigenetics, measured species kinetics and experimentally validated stimulus/reward biology are not resolved.
