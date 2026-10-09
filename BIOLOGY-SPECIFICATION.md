# MYCELIA mechanistic biology specification

Revision 1 — 2026-10-02. This document is the contract for a new core, not a claim that all mechanisms below exist in code.

## Objective and biological scope

Model fungal physiology through explicit local material, water, pressure and energy states, so larger behavior follows from their coupling. Each mechanism must have a biological rationale, a dimensional law, owned state, a numerical contract, observable predictions and a validation status. Implementation coverage, numerical correctness and empirical agreement are separate assessments. There is no defensible single biological-fidelity percentage.

The former `mycelia.organism.Mycelium` is retained as a **legacy prototype**, including its saved experiments. The independent core is `mycelia.biology.Hypha`. The new core currently resolves a straight, compartmented hypha with one prescribed apex and finite local baths. It does not yet model a branching organism.

**Species policy:** the later network benchmark remains *Phanerochaete velutina*. Cellular examples below come from *Neurospora crassa* and *Phycomyces blakesleeanus*; they motivate mechanisms, not a mixed-species calibrated organism. A parameter set cannot be named after a species until its strain, stage, medium, temperature, measured parameters and uncertainty are recorded. The current parameter set is illustrative and uncalibrated.

## Milestone 1: implemented causal chain

```text
finite bath substrate → membrane uptake → soluble carbon concentration
→ osmotic driving pressure → water exchange and septal flow → turgor
→ apical wall yield + finite vesicle cargo + ATP equivalents
→ permanent wall expansion → reduced pressure → replenishing flow
```

All rules operate on local compartments, membrane faces or septal edges. The global pressure solve enforces coupled conservation; it does not choose growth destinations or allocate resources by an optimization reward. No neural network, image classifier, reward label or associative image memory participates in this engine.

Wall expansion and osmotic water uptake have experimental biophysical support; the specific reduced constitutive laws and their coupling here are modeling choices. [R1](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2012.00099/full) examines irreversible deformation localized to a growth zone as well as elastic deformation. We use a finite growth-zone length rather than multiplying extension by the entire growing compartment's length. No sporangiophore constants from that study are copied as hyphal calibration.

The wall cargo is represented by two finite material pools: cytoplasmic cargo and an apical vesicle-supply reservoir. This is a lumped analogue, not a molecular Spitzenkörper. [R3](https://pubmed.ncbi.nlm.nih.gov/24523289/) concerns exocyst/Spitzenkörper function in *N. crassa*; it does not establish our first-order transport rates.

## State ownership and dimensions

| Owner | State | Units | Meaning / reduction |
|---|---|---|---|
| Parameters | temperature | K | Enters ideal dilute osmotic law; other rates are not temperature calibrated |
| Local bath | water | pL | Finite external water, shared by every cell referencing that bath |
| Local bath | nutrient | pmol monomer equivalents | One effective permeant carbon-bearing solute |
| Local bath | inert osmolyte | pmol particles | Impermeant, non-carbon external osmotic challenge |
| Compartment | rest volume `V0` | pL | Relaxed wall-enclosed volume; plastic expansion changes this |
| Compartment | water `W` | pL | Incompressible intracellular fluid volume; excludes resolved dry-material volume |
| Compartment | nutrient `N`, osmolyte `O` | pmol monomer equivalents | One osmotic particle per equivalent, an explicitly coarse approximation |
| Compartment | reserve `S` | pmol monomer equivalents | Stored polymer; excluded from osmotic particle count |
| Compartment | wall `B` | pmol monomer equivalents | Inserted structural material, excluded from osmotic count |
| Compartment | energy `E` | pmol ATP equivalents | Bookkeeping pool, not molecular ATP/ADP/Pi chemistry or free energy |
| Tip compartment | cytoplasmic cargo `C`, apical cargo `A` | pmol monomer equivalents | Finite wall precursor payload; not vesicle counts |
| Septum | effective opening | dimensionless 0–1 | Prescribed conductance fraction, not damage-responsive gating yet |
| Septum | signed last flow | pL/min | Positive from endpoint `a` to `b` |
| Derived | concentration | pmol/pL = mol/L | Concentration is recomputed from amounts and water |
| Derived | turgor/osmotic pressure | MPa | Nonnegative turgor, pressure relative to external hydrostatic pressure zero |
| Derived | rest length | µm | `V0/(πr² × 0.001)` with fixed cylinder radius |
| Ledger | respiration, ATP use, salt addition | corresponding amounts | Explicit sinks or external interventions |

Conversion constants: `1 µm³ = 0.001 pL`; `1 pL = 10^-15 m³`; `1 pmol/pL = 1 M`. Ideal osmotic `R = 0.008314462618 MPa L/(mol K)`. Every supplied kinetic/geometry parameter is in `biology/state.py` with its unit. Values are illustrative; finite seed pools are part of every budget.

The model does not resolve water-bound macromolecules, solute activity coefficients, protein crowding, ATP's osmotic contribution, distinct wall polymers, membrane area expansion costs, cap curvature or vacuolar fluid. Quantitative fitting would have to test these approximations.

## Equations implemented

### Uptake and metabolism

Local uptake demand is `U = vmax × membrane_area × c_external/(Km + c_external)`. All cells sharing a bath are limited together by its available nutrient, so substrate removed equals intracellular substrate added.

Reserve mobilization and respiration are first-order reactions integrated as finite fractions `1-exp(-k Δt)`. Reserve becomes soluble carbon. Respired carbon enters a cumulative sink and generates `yield × respiration` ATP equivalents. Maintenance spends at most the available ATP. Carbon yield is not a complete stoichiometric reaction model.

Cargo synthesis transfers soluble carbon to `C` and spends ATP. Motor delivery transfers `C` to `A`, also spending ATP. These are effective first-order reactions. There are no modeled actin/microtubule tracks, motor proteins or vesicle membrane lipids.

### Osmotic and elastic state

`π_internal = R T (N + O)/W`; `π_external = R T (bath_nutrient + bath_inert_osmolyte)/bath_water`.

`P = K × max(W/V0 - 1, 0)`.

Non-growing wall deformation is reversible; plastic growth changes `V0` separately. If water falls below `V0`, turgor is zero rather than becoming negative. The model then has a deflated fluid compartment; membrane detachment and collapse geometry are not resolved. The relaxed geometry must not be interpreted as measured wet hyphal diameter/length under shock.

Membrane water demand is `J = Lp × area × (π_internal - π_external - P)` with external hydrostatic pressure zero. Its sign can reverse. It transfers water only, not bath salt or cargo.

### Coupled pressure flow

For effective septal pores, `Q_ab = G_ab(P_a-P_b)`, where

`G = opening × 60000π r_pore^4/(8 μ ℓ_septum)` in pL/(MPa min), with pore dimensions in µm and viscosity in Pa·s. This is a reduced hydraulic constitutive law, not a resolved flow field. Cytoplasmic viscosity and pore geometry need measurement.

For a substep, membrane osmotic pressures are lagged and elastic pressure is implicit. Let `C_i = V0_i/K`, `M_i = Lp area_i`, and `L_G` be the conductance Laplacian:

```text
A = diag(C/Δt + M) + L_G
b = (W_old - V0)/Δt + M (π_internal_old - π_external_old)
P >= 0, A P - b >= 0, P · (A P - b) = 0
```

An active-set solver enforces the unilateral wall relation. With positive turgor this reduces to `A P = b`. Actual fluxes share donor budgets. A donor transfers at most 10% of its water per substep as a numerical safeguard, counted in `limiter_events`. This safeguard must remain inactive for a claimed refinement comparison; all milestone validation arms report zero activations.

Soluble carbon/osmolyte advects only across septa at the donor's pre-transfer concentration. Septal Fick diffusion uses `D × pore_area/ℓ × concentration_difference`, converted to pL/min, and shared solute donor limits. ATP, stored polymer and vesicle cargo do not freely cross septa in this reduction.

[R4](https://pubmed.ncbi.nlm.nih.gov/20538649/) motivates coupling growth to replenishing flow in *P. velutina*. Our engine evolves pressure from water and wall state rather than fitting observed cord currents; that paper does not validate this engine's rates.

### Polar wall expansion

The apical growth-zone rest volume is `Vg = πr² × min(growth_zone_length, rest_length) × 0.001`.

Mechanical expansion demand is `ΔV_mech = Vg φ max(P-Y, 0) Δt`. For a fixed radius, inserted wall carbon per volume is `ρV = 2000 ρwall/r` pmol/pL. Insertion capacity is `A × (1-exp(-k_exocytosis Δt))`; available ATP imposes a further limit. The actual inserted carbon is the minimum of mechanical demand, cargo insertion capacity and energy capacity.

Actual insertion transfers material from `A` to wall `B` and increases `V0` by `inserted_carbon/ρV`. **No water is created or reassigned to growth.** With water unchanged, expansion relieves turgor, inducing flow on the next substep. The positive feedback is therefore mediated by state and conserved fluid, not a tip-directed flow reward.

A tip reaching twice the initial segment length splits conservatively into two equal rest volumes. Soluble pools, water, reserve, wall and ATP divide proportionally; the apical cargo pools follow the new tip. A new septal edge appears. This is an explicit discretization/septation assumption: ring assembly, septum synthesis cost and cell-cycle control are not yet modeled. It must not be called emergent branching.

## Algorithm and interfaces

Each public step validates the state and snapshots it. Each substep performs: collectively bounded uptake → local reactions/cargo transport → coupled water solve and conservative advection → solute diffusion → funded tip wall expansion → conservative segment split → budget/state validation. A failed substep rolls back the entire public step and raises the error.

The default internal maximum is 0.01 min, public observation interval 0.1 min. Internal refinement tests compare 0.02, 0.01 and 0.0025 min; lagged osmotic concentration and operator splitting mean the method is not exact. Pressure uses a dense matrix in this small milestone: memory/time do not scale to a large branching network. The compartment cap is 128. It prevents further splitting but does not stop continuous elongation, so tip resolution degrades at the cap; do not claim validated runs beyond that regime.

`Hypha.run(duration_min, dt_min)`, `step(dt_min)`, `hyperosmotic_shock(delta_MPa)`, `save(path)` and `load(path)` are public interfaces. Units are explicit in names. Checkpoints use `mycelia.mechanistic-hypha.v1`; prototype files are rejected, because their amounts and time units have different meanings. Parameter changes to a saved run are not supported implicitly.

## Conservation contracts

Carbon equals bath substrate + intracellular soluble/storage/wall/cargo pools + cumulative respired carbon. Recorded external nutrient additions/removals extend the carbon ledger through `set_bath_nutrient(bath_id, amount_pmol)`. Unbudgeted direct nutrient edits are still rejected. Total water includes all baths and cells and remains fixed. ATP equals initial ATP + yield × respired carbon − cumulative ATP use. Inert bath osmolyte equals initial salt + recorded shock addition.

These are lumped accounting identities, not elemental/charge/thermodynamic conservation across a full metabolic network. In particular, ATP equivalents do not carry separate carbon atoms. Physical-ion extensions must add charge, elemental stoichiometry and free-energy accounting before making those stronger claims.

## Validation gates and outcomes

Run `python -m mycelia.biology --validate --output runs/biology-validation`. The protocol uses matched finite seed resources, zero initial ATP/cargo, basal-only external nutrient, and 30 min of evolution. Wall creep, motor delivery, exocytosis and ATP regeneration are individually disabled. Water exchange and septal continuity are separately disabled, with residual growth expected from stored material/initial strain. A +0.6 MPa bath shock is imposed at minute 10. A separate isolated assay starts an equal-pressure pair with tip cargo and suppresses uptake, metabolism, membrane water and diffusion to identify growth-induced flow.

Numerical tests additionally compare an analytic two-compartment hydraulic relaxation, closed septa, finite shared baths, nonnegative turgor, segment material partition, state restart, rejected unbudgeted edits, and rollback after an injected solver failure. Result JSON states computational pass/fail and explicitly records empirical validation as **not performed**.

[R2](https://www.microbiologyresearch.org/content/journal/micro/10.1099/mic.0.023507-0) reports *N. crassa* water/turgor loss and arrested growth after hyperosmotic shock, followed by ion/glycerol-dependent recovery. This milestone tests only the initial physical response and deliberately lacks regulatory recovery. Qualitative similarity of those initial signs is not reproduction of the time courses.

## Five-scale architecture and next gates

| Scale | Future owned state | Required local coupling | Admission gate |
|---|---|---|---|
| Molecular | named ions, neutral solutes, ATP/ADP/Pi, channel gates, membrane voltage, Ca buffers, diffusible signals | ion chemical/electrical potential → channel currents/pumps → osmotic pressure and polarity | species/strain channel identities, charge/stoichiometric balance, current/voltage clamp or flux observations |
| Compartment | cytoplasm and vacuole volumes, membrane area, wall constitutive parameters, septal pore/damage/repair state | membrane/wall/vacuole flux → pressure and compartment homeostasis | pressure/volume shock time courses, wall creep, septal closure and recovery experiments |
| Hypha | spatial polarity machinery, cargo localization, track occupancy, wall age/stiffening, nascent branch domains | local signaling/transport → spatial wall yield and expansion; local damage → repair | vesicle localization plus extension observations and perturbations; branch latency/location distributions |
| Network | spatial growing geometry, recognition/contact states, differentiated conduits, transport allocation, recycled material | local extension/contact → topology; pressure/material histories → remodeling | live network time series, tracer transport, fusion frequencies, cord mass and pruning data |
| Organism | no central controller; observables computed from lower states | persistent morphology/chemistry and environmental interactions → foraging/adaptation | held-out environmental changes, controls that remove candidate memory without geometry/resource confounds |

Future equations are **candidates until evidence and parameters are specified**: membrane charge balance `Cm dVm/dt = -ΣIi`; conservative ionic electrochemical transport; buffered Ca reaction-diffusion; vacuolar membrane exchange; reaction-diffusion or particle polarity dynamics; local wall maturation; local recognition/adhesion/pore-opening sequence; material-funded conduit differentiation. No species-specific channel kinetics, Cdc42 oscillator or electrical-memory law is silently invented in this revision.

The separate graph prototype now permits a starved tip state to reactivate, or a viable existing compartment to initiate a new tip, when that living tissue has sufficient local nutrient and energy and substrate exceeds `Config.resprout_threshold`. These are bounded recovery hypotheses added to prevent irreversible simulation lock-up; they are not fitted to time-lapse fungal data and do not establish biological resprouting fidelity. Extending a tip still pays normal carbon, water, and energy costs.

Milestone 2 should add a species-specific osmoadaptation model with at least ions, neutral osmolyte production and membrane energetics, then test complete pressure/volume/growth recovery data under multiple shock sizes. Failure to recover is visible in milestone 1, not tuned away. Obtain measured time-series data and separate calibration from held-out shocks before reporting biological agreement.

Milestone 3 should resolve apical polarity/cargo/geometry and local branch initiation, then add fusion with recognition and membrane continuity. Milestone 4 adds cord differentiation, damage/repair, environmental fields and morphology/tracer benchmarks. Milestone 5 revisits ecological memory only after a colony can actually reach/colonize resource baits; the legacy directional-memory experiment failed its primary control comparison and had no central bait contact. Its image prototype recall is not empirical fungal memory.

## Literature register

- **R1:** Ortega et al. (2012), *Phycomyces* growth-zone mechanics, [primary article](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2012.00099/full). Used for model structure, not default values.
- **R2:** Lew & Nasserifar (2009), *N. crassa* hyperosmotic responses, [primary article/abstract](https://www.microbiologyresearch.org/content/journal/micro/10.1099/mic.0.023507-0). Initial response and recovery are separate targets.
- **R3:** Riquelme et al. (2014), *N. crassa* exocyst/Spitzenkörper, [primary record](https://pubmed.ncbi.nlm.nih.gov/24523289/). Molecular assembly is not reproduced by our two cargo pools.
- **R4:** Heaton et al. (2010), growth-induced flow in *P. velutina*, [primary record](https://pubmed.ncbi.nlm.nih.gov/20538649/). Network-scale motivation, not cellular calibration.
- **R5:** Fukasawa et al. (2020), relocation/ecological memory, [primary record](https://pubmed.ncbi.nlm.nih.gov/31628441/). Benchmark for a later spatial colonization model; not passed by the current prototype.

The pasted design brief's `chatgpt-content-reference` markers are not resolvable citations. This register replaces them with identified papers. Other matrix entries explicitly need further primary-literature review before implementation.
