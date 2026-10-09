# MYCELIA biological fidelity matrix

This matrix separates implemented state/laws from hypotheses, numerical checks and empirical evidence. **All new-core rows remain empirically unvalidated.** Implementation is not proof of biological correctness; no percentages or progress bars are assigned.

The legacy prototype is preserved for comparison. The new core is the dimensional `mycelia.biology` milestone. Source IDs and scope restrictions are defined in [BIOLOGY-SPECIFICATION.md](BIOLOGY-SPECIFICATION.md). “Needs literature” is an explicit evidence gap, not a claim that biology is unknown.

Each row identifies a target observable. Admission to an empirically validated status requires measured species/strain data, uncertainty, independently selected test data, error metrics and documented scope. Conservation/unit tests alone cannot change that status.

| Scale / mechanism | Legacy prototype | New core | Assessment | Next empirical target | Evidence |
|---|---|---|---|---|---|
| Molecular / Named ions | Absent | Absent | Not specified | Ion flux, concentration and charge balance | Needs species-specific literature |
| Molecular / Organic osmolytes | Lumped organic pool | Conserved single solute pool; no synthesis regulation | Reduction | Osmolyte accumulation after shock | R2 |
| Molecular / Metabolites | Lumped nutrient/energy | Single carbon substrate + ATP equivalents | Reduction | Substrate, respiration and metabolite time courses | Needs stoichiometric model |
| Molecular / Ion channels | Absent | Absent | Not specified | Current/voltage dependence and knockouts | R2 motivates next gate |
| Molecular / Membrane potential | Absent | Absent | Not specified | Voltage and input conductance after perturbation | R2 motivates next gate |
| Molecular / Calcium gradients | Absent | Absent | Not specified | Spatial Ca signal and tip extension after buffering | Needs species-specific literature |
| Molecular / Molecular signalling | Directional trace hypothesis | Absent | Not specified | Signal production, propagation and pathway perturbation | Needs species-specific literature |
| Compartment / Hyphal compartments | Explicit graph | Explicit serial cells with split rule | Implementation / geometry reduction | Spatial compartment lengths and septation times | Needs septation literature |
| Compartment / Cell wall | Elastic volume + growth proxy | Separate elastic state and funded irreversible expansion in fixed tip growth zone | Constitutive candidate | Pressure-step creep and extension response | R1 |
| Compartment / Plasma membrane | Water/uptake flux proxies | Finite local uptake and osmotic water flux; no dynamic voltage or membrane chemistry | Reduction | Selective permeability, uptake and osmotic response | R2 |
| Compartment / Septa | Pores and heuristic damage gate | Effective pore hydraulics/diffusion; prescribed gate; splitting not costed | Reduction | Pore conductance, closure/reopening and injury responses | Needs species-specific literature |
| Compartment / Cytoplasm | Lumped water/resources | Finite water and soluble material per cell | Reduction | Cytoplasmic tracer dilution and volume | R2 / R4 |
| Compartment / Vacuole | Absent | Absent | Not specified | Vacuolar volume and tonoplast flux during shock | Needs species-specific literature |
| Compartment / Turgor | Nondimensional elastic pressure | Nonnegative pressure from water/rest-volume wall law | Constitutive candidate | Pressure-probe traces under multiple stresses | R1 / R2 |
| Compartment / Water uptake | Finite osmotic proxy | Dimensional osmotic membrane exchange with finite baths | Physical constraint + constitutive candidate | Water uptake, shrinkage and refill time courses | R2 |
| Compartment / Nutrient uptake | Finite credited uptake | Local Michaelis-Menten capacity, shared finite bath budgets | Constitutive candidate | Species/substrate uptake saturation curves | Needs transporter measurements |
| Compartment / Metabolism | Fuel/reserve and ATP-like energy | Conservative reserve mobilization, respiration and ATP-equivalent accounting | Reduction | Oxygen/carbon flux and ATP/ADP balance | Needs metabolic stoichiometry |
| Compartment / Storage | Finite reserve pool | Finite non-osmotic polymer-equivalent reserve with release | Reduction | Storage pool size and mobilization kinetics | Needs species-specific literature |
| Hypha / Tip growth | Heuristic pressure/energy/vesicle factor | Local wall yield limited by cargo and energy; one prescribed apex | Constitutive candidate | Tip extension versus pressure, wall and transport perturbations | R1 / R3 |
| Hypha / Polarity | Tip direction / optional history bias | Fixed straight axis and tip flag; dynamic polarity absent | Prescribed / not emergent | Polarity localization, relocation and response times | Needs species-specific literature |
| Hypha / Spitzenkörper | Energy-dependent supply proxy | Finite apical wall-cargo reservoir; molecular organization absent | Lumped analogue | Vesicle localization and tip-shape response | R3 |
| Hypha / Vesicle transport | Supply proxy | Finite cytoplasmic-to-apical cargo delivery with ATP expense | Constitutive candidate | Cargo velocity, motor inhibition and localization | R3; rates uncalibrated |
| Hypha / Cytoskeleton | Absent | Absent; motor rate is a coarse substitute | Not specified | Track organization and motor/track perturbations | Needs primary literature |
| Hypha / Branching | Seeded local hazard | Absent; serial splitting is not branching | Not specified | Branch site/latency distribution with local precursor dynamics | Needs species-specific literature |
| Hypha / Avoidance | Local direction/resistance rule | Absent | Not specified | Obstacle/stimulus-specific turn and extension response | Needs species-specific literature |
| Hypha / Damage | Stress proxy / simplified healing | Absent | Not specified | Injury localization, leakage and survival | Needs species-specific literature |
| Hypha / Repair | Heuristic relaxation/recycling | Absent | Not specified | Material-funded wall/pore repair and recovery | Needs species-specific literature |
| Network / Anastomosis / fusion | Contact heuristic graph edge | Absent; recognition/contact/membrane continuity not modeled | Not specified | Compatible recognition, contact latency and cytoplasmic exchange | Needs primary literature |
| Network / Diffusion | Conservative substrate and graph diffusion | Fick transport across septal pores with donor limits | Physical law + geometric reduction | Tracer spread and measured diffusivity | Species coefficients needed |
| Network / Cytoplasmic streaming | Hydraulics plus tip-directed cargo proxy | Pressure-driven axial bulk flow; no resolved intracellular velocity field or active streaming | Reduction | Tracer speeds and radial flow profiles | R4; transport modes need review |
| Network / Growth-induced flow | Coupled water/growth but extension proxy | Wall expansion lowers pressure, generating flow in an isolated equilibrium-pair test | Computationally verified causal coupling | Simultaneous growth volume and tracer flux time series | R4 |
| Network / Resource redistribution | Conservative graph flows | Soluble advection/diffusion across a serial septal graph | Reduction | Basal-to-tip labelled substrate transport | R4; substance-specific data needed |
| Network / Topology | Growth/branch/fusion graph | Serial graph extends by prescribed conservative segment splits | Geometry reduction | Spatial morphology and topology through time | Branch/fusion dynamics deferred |
| Network / Cords / rhizomorphs | Effective conduit reinforcement | Absent; hypha is not a cord | Not specified | Differentiation anatomy, conductive area and cord biomass | R4 motivates later network gate |
| Network / Reinforcement | Flow-linked material-funded radius rule | Absent | Not specified | Flow history versus subsequent material investment | R4 |
| Network / Pruning | Dead-terminal retraction heuristic | Absent | Not specified | Carbon recycling and network retraction time courses | Needs primary remodeling data |
| Organism / Structural memory | Persistent graph; no validated memory assay | Persistent grown rest volumes; no functional memory assay | Persistence only | Transfer/regrowth with geometry and resource controls | R5 |
| Organism / Chemical memory | Engineered image receptor traces; directional hypothesis | No memory law; persistent pools are ordinary material state | Not demonstrated | Exposure-specific persistence and causal erase intervention | R5 does not identify molecular mechanism |
| Organism / Electrical activity | Absent | Absent | Not specified | Voltage/current recordings with artifact controls | R2; long-distance function requires separate evidence |
| Organism / Electrical memory | Absent | Absent | Unknown mechanism / not specified | Persistent state with electrical causal controls | No established model selected |
| Organism / Environmental memory | Directional hypothesis failed primary test | Absent; shock response has no regulatory recovery | Not demonstrated | Bait colonization and relocation versus matched controls | R5 |
| Organism / Sensing / adaptation | Gradient and stress proxies | Concentration drives uptake/osmotic exchange; active adaptation absent | Physical response only | Held-out stress dose/time and recovery response | R2 |
| Organism / Foraging | Resource-gradient growth heuristic | Absent in straight-axis core | Not specified | Actual bait contact/colonization and transport cost | R5 |
| Organism / Resource allocation | Local growth/transport heuristics | Local resource spending and pressure transport; no full colony allocation | Partial local coupling | Growth and carbon distribution after spatial interventions | R4 |

44 mechanisms are tracked at this revision, including every mechanism in the supplied design brief. CSV and JSON mirrors are provided for later updates. This is a coverage inventory, not a numerical fidelity ranking.

## Current admission gates

1. **Computational gate:** dimensional equations, finite-pool conservation, analytic checks, mechanism perturbations, checkpoint/rollback and timestep refinement. Implemented for the first coupled physiology slice.
2. **Cellular evidence gate:** selected species, measured uptake/wall/permeability/transport values and complete osmotic response curves. Open.
3. **Hyphal/network gate:** dynamic polarity, branching, fusion and remodeling with morphology/tracer validation. Open.
4. **Organism/memory gate:** actual resource colonization and matched relocation/erase controls. Open.

The earlier directional-memory report remains a negative result. The separate image memorization prototype is an engineered classification experiment and does not count as validation of biological chemical memory.
