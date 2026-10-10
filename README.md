# MYCELIA

## Mechanistic biology rebuild

The new core is `mycelia.biology`: a dimensional, conservative hypha model coupling finite nutrient uptake, osmotic water exchange, unilateral turgor, septal pressure flow, finite vesicle cargo, and energy-funded apical wall expansion. It is the first implemented milestone of the five-scale architecture in [BIOLOGY-SPECIFICATION.md](BIOLOGY-SPECIFICATION.md), with honest coverage in [BIOLOGICAL-FIDELITY.md](BIOLOGICAL-FIDELITY.md). Parameters are uncalibrated; quantitative biological validation has not been performed.

```bash
python -m mycelia.biology --duration 30 --output runs/biology
python -m mycelia.biology --validate --output runs/biology-validation
python -m mycelia.biology --resume runs/biology/state.json --duration 10 --output runs/biology-resumed
```

Install with the existing `pip install -e ".[dev]"` command below. Open `report.html` in the selected run folder; CSV time series and restart states are included. Only NumPy is needed for the core and report. To add scientific time-course plots after a validation run, install the optional `.[plots]` extra and run `python tools/plot_biology_results.py runs/biology-validation`. No classifier, CNN or neural network is used in these physiology experiments.

This milestone has one straight apex, effective cargo transport and prescribed septal gates. Explicit ions/calcium/channels, membrane voltage, vacuoles, molecular polarity, branching/fusion and osmotic adaptation are future gates, not completed mechanisms. The osmotic-shock assay tests water/turgor loss and growth arrest; recovery is absent.

## Test the new core with brain images

```bash
python -m pip install -e '.[dev,images]'
python tools/medical_image_test.py --dataset Brain-Tumor-MRI-Dataset --output runs/medical-images
python tools/predict_medical_image.py /path/to/image.jpg --readout runs/medical-images/external-physiology-readout.json
```

The first command processes all raw images through the dimensional core, then evaluates a separate Training-only NumPy nearest-centroid readout on exact-deduplicated Testing images. The new core is **not trained**. No CNN, neural network, SVM or sklearn is used in this test. Fixed image encoding, matched controls, readout baselines, patient/provenance limits and reproduction are documented in [MEDICAL-IMAGES.md](MEDICAL-IMAGES.md). Source archives omit the dataset; supply its folder path if it is stored elsewhere.

## Learn visual patterns without class labels

The experimental `VisualMycelium` path learns resource-paid local patch imprints
from 32x32 intensity, edge, and texture cues. Novelty recruits colony specialists;
reconstruction errors control competitive updates. It accepts no tumor category or
image ID, and uses no CNN, neural network, or external trained classifier.

```bash
python -m pip install -e '.[dev,images]'
python tools/unlabeled_visual_test.py --dataset Brain-Tumor-MRI-Dataset --output runs/visual-learning
python tools/recall_visual_image.py --model runs/visual-learning/unlabeled-visual-memory.json --image /path/to/image.jpg --output runs/visual-learning/recall.png
```

Reconstruction and source-identity retention measure visual memory; they are not
medical class accuracy. This remains a graph-memory prototype, separate from the
dimensional growing hypha. Protocol, mechanisms, and limits are in
[VISUAL-LEARNING.md](VISUAL-LEARNING.md).

The current full-data learner defaults to fixed full-frame preprocessing,
multi-scale cues, and a learned pair-motif context layer. On two strict masked
held-out splits, that layer improved grayscale patch prediction over
first-order context; a subsequent full pass processed all 7,200 available
images. This is still image-level context transfer and seen-image memory—not
tumor understanding or diagnosis. Run protocols and limitations are in
`VISUAL-LEARNING.md` and `ARCHITECTURE-DEBUG-REPORT.md`.

## Research tumor prediction from OMNIA images

The label-free visual learner now accepts full-resolution one-image OMNIA
containers through the user's SDK. A separate Training-label nearest-centroid
readout maps its frozen colony-match features to the dataset's four classes.
Training-only selection chose 1,024 colonies; held-out image accuracy was
66.76%, compared with 63.22% for a centroid using the same raw input cues.
The learner processed all 5,410 clean Training images; Testing contained 1,468
images after exact duplicate filtering. This is image-level research, with
distance scores rather than probability estimates.

See [OMNIA-TUMOR-RESEARCH.md](OMNIA-TUMOR-RESEARCH.md) for the protocol, commands,
resource use, evidence, and limits. `tools/omnia_tumor_research.py` trains and
evaluates; `tools/predict_omnia_tumor_research.py` predicts one `.omnia` image
using the saved model and preprocessing. The learning and readout use no CNN
or neural network.

The next-phase audit found that a matched conventional non-neural HOG/SVM
baseline reaches 89.58% on the same already-evaluated image split, while the
current MYCELIA readout reaches 66.76%. Exact full-source reconstruction is
also impossible from the current 32×32 cue bottleneck. The measured gaps,
few-label curve, compute costs, biological validation datasets, and admission
gates for a multi-resolution patient-linked architecture are documented in
[NEXT-MEDICAL-ARCHITECTURE.md](NEXT-MEDICAL-ARCHITECTURE.md).

Frozen inference can use an exact compact graph snapshot:

```bash
python tools/compile_omnia_visual.py \
  --memory runs/omnia-tumor-research/unsupervised-mycelia-memory.json \
  --output runs/omnia-tumor-research/frozen-graph.npz
python tools/predict_omnia_tumor_research.py /path/to/image.omnia \
  --model-dir runs/omnia-tumor-research --omnia-sdk /path/to/omnia-sdk \
  --compiled-model runs/omnia-tumor-research/frozen-graph.npz
```

The compact snapshot preserves scores and predictions; it improves deployment
cost only and does not change learning or accuracy.

## Patient-separated 3D MRI learning pilot

`tools/rhuh_mri_pilot.py` is an exploratory multimodal NIfTI experiment for the
RHUH-GBM research dataset. It keeps patient visits together, checks duplicate
downsampled inputs across the split, fits an unlabeled prototype bank on the
training patients, and predicts a withheld MRI patch from its six spatial
neighbors and the other center modalities. Expert masks are used only after
learning for a post-hoc cluster-alignment score. This is not a tumor diagnosis
tool, and the single-cohort results are not medical validation.

The pilot supports both the full five-sequence NIfTI layout and an indexed
four-sequence derivative. `tools/fetch_rhuh_public_nifti.py` fetches the
image/mask files only from a pinned 28-patient CC BY 4.0 derivative revision and
verifies every file hash. It excludes restricted DICOM and visualization-only
meshes. This mirror omits ADC and 12 patients from the source's 40-patient
cohort; the experiment summary records the exact revision and used sequences.
The loader aligns each expert mask to its T1-contrast grid with nearest-neighbor
resampling, preserving discrete labels.

The five-fold 28-patient run showed low masked-sequence RMSE, but that predictor
is a conventional Ridge model with MiniBatchKMeans features, not the native
fungal growth graph. Unsupervised cluster alignment to tumor subregions was
weak (mean ARI 0.022); its coarse patch readout missed necrosis and enhancing
tumor. Full metrics and the architecture gap are documented in the
`PATIENT-HELD-OUT-3D-RESULTS.md` output report.

The pilot also has an opt-in `--visual-representation dct8` ablation. It
encodes each 4×4×4 patch as eight signed low-frequency DCT features so spatial
arrangement is retained beyond mean/texture/gradient summaries. Five-fold
patient-held-out testing found sequence-dependent results: MYCELIA improved
the pooled low-frequency patch projection over Ridge by 4.2% MSE, while mean
normalized coefficient error was slightly worse overall. This is a partial
projection, not full-resolution image reconstruction or tumor diagnosis; see
`DCT8-PATIENT-TRANSFER-RESULTS.md` in the phase output directory.

An audit found that per-colony weights were transferred by colony index between
independently fit models, even though those IDs can refer to different visual
patterns. The earlier per-colony result is superseded and should not be used.
Prototype alignment is now implemented and tested. A new all-64-mode 3D patch
experiment found no improvement over the matched shared Ridge predictor; its
calibrated colony specialists fell back to shared prediction in nearly every
fold. The full detail result is in `rhuh-dct64-full-patient-cv-aligned/` in the
phase output directory. The prior group-level sequence-gated DCT8 result is
unaffected, but full-detail learning still needs a more effective local growth
and specialization rule.

```bash
python -m pip install -e '.[medical]'
python tools/fetch_rhuh_public_nifti.py --output /path/to/RHUH-GBM-derived-28patients
for fold in 0 1 2 3 4; do
  python tools/rhuh_mri_pilot.py --dataset /path/to/RHUH-GBM-derived-28patients --output runs/rhuh-pilot/fold-$fold --seed 20261014 --fold $fold --folds 5 --modalities auto
done
python tools/summarize_rhuh_cv.py --fold-root runs/rhuh-pilot --output runs/rhuh-pilot/cross-validation-summary.json
```

The public RHUH-GBM release is hosted by TCIA under CC BY 4.0; cite the dataset
DOI `10.7937/4545-c905`. Its DICOM package contains controlled-access material;
this pilot uses the public NIfTI release. Dataset files are not included in the
source archive.

For the local TIFF subset, `tools/tcga_lgg_slice_context_test.py` learns a
separate six-direction colony-transition memory across consecutive slice
indices, while preserving the patient-level outer split and masking each target
patch before assignment. Its route/local-evidence blend weight is selected on
a patient-disjoint inner validation split within each outer training fold.
Since that subset has no physical slice spacing or
orientation metadata, this is 2.5D ordered-slice context—not validated 3D
geometry. The nested blend slightly improved over a six-neighbor pixel-mean
baseline in this run, but its patient-bootstrap interval crossed zero, so the
evidence remains inconclusive. Run it with:

```bash
PYTHONPATH=.:tools python tools/tcga_lgg_slice_context_test.py \
  --dataset /path/to/brain_tiff --output runs/tcga-slice-context --folds 5
```

## Legacy prototype

The following `python -m mycelia` and `main.py` commands still run the earlier nondimensional graph prototype. Its MRI and memory experiments remain separate. Neither its accuracy nor its heuristic branching validates the new mechanistic engine. Prototype and mechanistic checkpoints have distinct schemas and cannot be exchanged implicitly.

A clean experimental mycelial simulation. One compartment graph connects local sensing, nutrient uptake, reserves, metabolism, osmotic water exchange, elastic pressure, hydraulic and solute transport, tip extension, branching, septation, fusion, and remodeling.

The package contains the simulation and its verification suite. It does not depend on the old MRI classifier, OMNIA SDK, neural checkpoints, or historical phase scripts.

## Run in VS Code

Extract `MYCELIA-clean.zip`, then open its **MYCELIA** folder in VS Code. Install the Microsoft Python extension if needed, open a terminal, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m mycelia --steps 300 --output runs/demo
```

On Windows, use `python -m venv .venv` and `.venv\Scripts\Activate.ps1` instead of the first two commands. Select `.venv` through **Python: Select Interpreter**; the supplied default interpreter setting targets macOS/Linux.

Open `runs/demo/report.html` in your browser. It contains the network, nutrient substrate, growth and flow charts, and budget diagnostics. `network.svg` can be exported without a plotting dependency.

`python main.py` is equivalent to `python -m mycelia`. The Run and Debug menu includes **MYCELIA: nutrient patches** and **MYCELIA: tests**.

## Other experiments

```bash
python -m mycelia --scenario uniform --steps 300 --seed 7 --output runs/uniform
python -m mycelia --scenario poor --steps 300 --output runs/poor
python -m mycelia --scenario obstacle --steps 300 --output runs/obstacle
python -m mycelia --config config.example.json --steps 600 --output runs/long
python -m mycelia --resume runs/demo/state.json --steps 100 --output runs/resumed
```

The default seed makes a run reproducible. Saving includes the environmental fields, graph, physiology, random generator, counters, and bounded histories. Loading restores the same evolution. A resumed run uses its saved configuration; `--dt` can change subsequent step duration. `--steps` is the number of additional steps.

Each run writes `summary.json`, `state.json`, `network.svg`, and `report.html`. Output paths are relative to your terminal directory. The CLI exits with a failure message on invalid configuration or numerical errors.

## Model and units

This is an **uncalibrated compartment-level research model**. Its nondimensional amounts, spatial grid units, and model-time units are explicit computational assumptions. Temperature uses Celsius and enters a phenomenological metabolic response centered at 25°C. None of the default parameter values is advertised as a measured species-specific constant.

Each node owns a finite hyphal compartment. Its rest volume is `π radius² length`; its dynamic water amount determines wall pressure `P = (water / rest_volume - 1) / wall_compliance`. Pressure relative to the relaxed wall may be negative; reported turgor is `max(P, 0)`.

External nutrient and water fields are **finite amounts**, not normalized image intensities. Substrate nutrients diffuse conservatively across cell faces with reflecting boundaries. Uptake credits only what was actually removed from the substrate. Nutrient, organic osmolyte, reserve, biomass, memory material, conduit investment, and respired carbon form the carbon ledger. ATP-like energy is produced from respired substrate with a fixed yield, spent on maintenance, growth and memory synthesis, and explicitly dissipated when dead tissue retracts. Seed material is included in each initial budget.

Osmotic potential is modeled as `osmotic_scale × solute / water`. Membrane flux responds to internal minus external osmotic potential minus wall pressure. Imported water is removed from the substrate; reverse flux is returned. Osmolytes are lumped organic-solute pools, not an explicit ion chemistry model.

Graph flow uses `Q = G(Pa-Pb)` with effective Poiseuille-style conductance and a regulated septal opening. The pressure solve is implicit:

```text
(compartment capacitance / dt + conductance Laplacian) P
    = (current water - rest volume) / dt
```

A matrix-free solver handles disconnected compartments without inventing balancing sources or sinks. Actual water and solute transfers share each donor's available pool; they cannot overspend a donor with several outgoing edges. Nutrients and osmolytes also diffuse. A separate, explicitly phenomenological nutrient-cargo term points toward active tip compartments and does not manufacture water.

Growth uses turgor, energy density, an energy-dependent vesicle-supply proxy, local substrate resistance, damage, and local tip avoidance. Unresolved tip elongation accumulates in an explicitly budgeted tip buffer until it reaches the configured compartment length (default 0.5 grid units). This avoids creating arbitrarily tiny compartments with vanishing uptake surfaces. Every elongation pays water, carbon, and energy costs; forming the new compartment transfers its buffered water/biomass and divides internal pools conservatively. Branch maturation has its own clock. Branching uses a seeded local hazard; new tips share the existing compartment until they extend. Each new compartment connection creates a septal pore. Stress closes pores and reduces growth; energy supports a simplified damage relaxation.

Fusion uses local spatial contact, same-organism living tissue, duplicate-edge protection, and an arc-length exclusion neighborhood. This prevents neighboring discretization points from being mistaken for a new fusion. Fusion creates cytoplasmic continuity without automatically erasing the growing front. Some scenarios may produce no fusion; there is no forced fusion quota.

Flow activity averages **absolute** throughput. Used conduits can receive nutrient-funded reinforcement. Starved tips deactivate; unused dead terminal tissue can retract, returning carbon and water to the substrate. Interior bridges are preserved by this pruning rule. Effective conduit radius describes pathway investment; it is not a separate resolved fluid volume. Existing nodal volumes hold the water.

## Mechanism coverage

| Layer | Included | Further work |
|---|---|---|
| Environment | finite nutrients/water, gradients, oxygen, temperature response, obstacles, stress/resistance | species-specific chemistry and substrate mechanics |
| Compartment physiology | nutrients, reserves, biomass, ATP-like energy, organic osmolytes, water, elastic pressure, damage | membrane channel kinetics, vacuoles, stoichiometric metabolism |
| Hypha | polarized extension, branching, lumped vesicle supply, compartment boundaries | cell-wall rheology, calcium dynamics, cytoskeleton, molecular Spitzenkörper |
| Network | hydraulic transport, solute advection/diffusion, cargo proxy, septal gates, fusion, reinforcement/retraction | species-calibrated conductance and repair mechanisms |
| Organism | emergent morphology and transport, persistent state, resource interventions | validated biochemical adaptation and material phenotypes |

The code does not claim complete biological replication, a biological accuracy percentage, fungal cognition, or clinically useful MRI prediction. Electrical signaling and explicit molecular pathways require separately specified and validated mechanisms before addition.

## Programmatic use

```python
import numpy as np
from mycelia import Environment, Mycelium, Config

environment = Environment(np.full((64, 64), 2.0), water=8.0)
organism = Mycelium(environment, Config(), seed=42, position=(20, 20))
organism.run(100)
organism.add_resource_patch(45, 40, radius=6, strength=3)
organism.deplete_region(20, 20, radius=4)
organism.run(100)
print(organism.summary())
organism.save("runs/custom/state.json")
```

Use the organism's resource-intervention methods to keep external carbon changes in the ledger. Oxygen, stress, resistance, and obstacle arrays can be changed directly before stepping. Directly editing nutrient/water pools without recording their budgets is rejected. Call `Environment.step()` directly only for standalone substrate experiments.

## Verification and limits

Tests check conservative diffusion, bounded actual uptake, simultaneous donor limits, pressure direction, closed septa, disconnected networks, growth/branching/transport integration, resource interventions, rich/poor response, obstacles, persistent fusion, retraction recycling, timestep validation, transactional rollback, state restart, and CLI output.

Carbon, water, and ATP-like budget checks execute after every public step. Failed steps restore graph, substrate, random generator, and ledgers, then raise the original error. No failures are silently turned into successful snapshots.

The default is `dt=0.5`, 800 stored compartments, and 48 active tips. Public steps are subdivided to at most 0.5 model-time units; environmental diffusion has its own positivity substeps. The implicit pressure solve avoids dense matrices. Limits include all stored nodes, including retired tissue, so the model stops adding compartments at the cap instead of growing memory indefinitely. Events retain the newest 3,000 entries and summaries the newest 1,000 steps; total event counts remain available. Large runs still incur transactional-copy, graph, and visualization costs.

The checks verify computational consistency and scenario behavior. Quantitative biological validation still requires a chosen fungal species, measured parameters, observable experimental targets, and calibration.

## Source layout

```text
mycelia/
  state.py          configuration, compartments, segments, tips
  environment.py    finite substrate fields and scenarios
  transport.py      pressure solve and conservative transfers
  organism.py       shared physiology/growth/fusion/remodeling loop
  report.py         portable SVG and HTML run report
  cli.py            command-line entry point
tests/              conservation, intervention, integration and CLI checks
.vscode/            launch and test settings
main.py             direct entry point
config.example.json parameter overrides
```

## Optional brain image learning test

The physiology simulator is not a classifier. A separate supervised readout can learn folder labels from image descriptors and/or simulated physiology. Install with `python -m pip install -e ".[learning]"`, then run:

```bash
python tools/accuracy_test.py --dataset Brain-Tumor-MRI-Dataset --output runs/brain-accuracy-test
```

The script extracts unsigned gradient histograms and pooled intensities plus 60-step physiology descriptors. It compares physiology-only, image-only and combined readouts using a fixed linear SVM. It fits scaling and classifiers using Training images only, removes exact decoded-image duplicates and train/test overlap, and records unseen-image accuracy, per-class recall, confusion matrices, individual predictions and saved models. Labels never enter the biological simulation.

This is an image-level experiment. Patient independence and near-duplicate separation cannot be established from the available folder names. Generated accuracy describes the supervised readout, not independent learning by the fungal simulation or diagnostic performance. Do not load joblib models from untrusted sources.

## Learning in the compartment network

The opt-in adaptation milestone adds fast/slow funded traces and cord chemistry that can update at maximum radius. Run `python tools/adaptation_test.py --output runs/adaptation` for paired A→B→A tests, or `python tools/adaptive_medical_validation.py --dataset Brain-Tumor-MRI-Dataset --output runs/adaptive-mri-validation` for a separate Training-only MRI comparison. See [ADAPTIVE-MEMORY.md](ADAPTIVE-MEMORY.md). This extends the prototype graph module; the dimensional physiology core remains separate.

To train and test the direct graph memory, then save a separate predictor fitted on **all** valid MRI images:

```bash
python -m pip install -e '.[images,dev]'
python tools/train_mycelium.py --dataset Brain-Tumor-MRI-Dataset --output runs/full-data-memory --epochs 2
python tools/predict_memory_image.py --memory runs/full-data-memory/full-data-memory.json --image /path/to/image.jpg
```

This selects capacity and optional exposure-dependent consolidation using a validation split from Training only. It then fits all clean Training images and tests a frozen model on unseen, deduplicated Testing images. Finally, a distinct model learns all original Training **and** Testing images. That final model's scores on these images measure training recall. See [FULL-DATA-TRAINING.md](FULL-DATA-TRAINING.md) for the protocol and [BIOLOGICAL-MEMORY.md](BIOLOGICAL-MEMORY.md) for the mechanism. Dataset images are supplied separately.

Version 1.2.1 makes scalar and batch predictions use the same label order when scores tie. It preserves the trained checkpoint contents. The measured retention limits and proposed next mechanisms are documented in [ADAPTATION-REVIEW.md](ADAPTATION-REVIEW.md); these proposed mechanisms have not yet been implemented.

`AssociativeMycelium` stores learned chemical imprints in actual `Mycelium` compartments and reinforced cords. The new experiment uses **no SVM, CNN, neural network or independently fitted readout**. Its algorithm is explicitly a bio-inspired online prototype memory, implemented in biological graph state; it is not evidence of fungal MRI cognition.

```bash
python -m pip install -e ".[images,dev]"
python tools/network_memory_test.py --dataset Brain-Tumor-MRI-Dataset --output runs/network-memory-test
python tools/predict_memory_image.py --memory runs/network-memory-test/network-memory.json --image /path/to/image.jpg
```

Teaching labels select the population receiving a nutrient reward. Local compartments synthesize slowly adapting receptor traces using real nutrient and ATP-like pools; successful local co-exposure reinforces cord material and actual hydraulic radius. Periodic physiological timesteps preserve osmotic flow, pore regulation, metabolism and resource budgets. Slow turnover recycles trace/cord material and degrades the imprint. Frozen recall compares incoming chemical cues with stored node and cord traces without altering the network.

This first experiment uses a deliberately pre-inoculated square lattice, with no growing tips, as a reproducible sensing scaffold. The free-growth simulator still supplies branching, fusion, vesicle polarity, damage and retraction; we do not claim the MRI experiment demonstrates learning through branching/fusion. Recognition is direct chemical-imprint matching across colony populations. The mapping from grayscale/gradient to chemical cues and from labels to rewards is an explicit computational assumption. It is similar mathematically to online prototype clustering, with nutrient-paid state changes and cord geometry.

See `BIOLOGICAL-MEMORY.md` for equations, biological evidence and scope. The older optional SVM experiment remains a clearly separate baseline.

## Phase 2: directional regrowth experiment

```bash
python tools/directional_memory_test.py --trials 30 --output runs/directional-memory-test
```

The opt-in candidate mechanism consolidates a locally sensed nutrient-gradient direction when the compartment actually takes up resource. Trace synthesis consumes nutrient and ATP-like energy; slow turnover returns its material to reserve; growing compartments inherit a diluted, conservatively transferred imprint. The trace can bias later tip direction. These rules are hypotheses, not measured fungal molecular kinetics. Default simulations leave acquisition and directional bias disabled.

The assay trains a growing network in a Gaussian resource field, then transfers only the root compartment into fresh uniform substrate, discards all old geometry/tips, regenerates tips with randomized polarity and disables further acquisition. Six paired arms compare intact memory, erased trace, directional coupling disabled, inheritance disabled, rotated trace and uniform-history sham. All physical initial pools are matched across arms.

**First result: the primary behavioral check did not pass.** In 30 paired trials, the memory-minus-erased alignment difference was +0.157, with a 95% bootstrap interval of −0.142 to +0.437. The direction initially follows the bait but is overwritten by subsequent local gradients. None of the networks reached the bait's central radius-3 region. This is a gradient-history regrowth assay, not a replication of the published wood-bait experiment. See `DIRECTIONAL-MEMORY.md` for the protocol, limitations and next model requirements.

## Receptor-field memory (new)

Unsupervised receptor growth + teach-only memory reaches **94.5%** held-out 4-class accuracy and **97.3%** tumor-presence accuracy on the brain-tumor MRI dataset (previous colony memory: 73.3%). See [RECEPTOR-MEMORY.md](RECEPTOR-MEMORY.md).

## Mycelial Symbiosis Engine (new)

A self-modifying trace memory with structural plasticity, local reward/punishment, a metabolic budget, fast/slow memory with sleep, and self-healing. 93.7% diagnosis and 97.4% tumor presence with 6x less memory than the fixed network; recovers to full accuracy after losing 50% of its memories. See [SYMBIOSIS-ENGINE.md](SYMBIOSIS-ENGINE.md).

## Collective Dissent (new)

Independent specialists plus a coordinator that abstains when evidence is unreliable: at 90% coverage, accuracy on answered scans rises from 93.7% to 98.1% (68 of 93 errors caught before output). See [COLLECTIVE-DISSENT.md](COLLECTIVE-DISSENT.md).

## Symbiosis Lab v0.1 (new)

A bounded self-improvement loop: detects a weakness, gathers evidence, tests candidates against frozen production, rejects regressions, verifies, and asks a separate gate for a one-time holdout score. Only a human can approve a release. See [SYMBIOSIS-LAB.md](SYMBIOSIS-LAB.md).

## MLCP — internal communication protocol (new)

Specialists exchange one 4-bit learned token each in a checksummed frame: same diagnosis accuracy as raw vectors with 2.9x fewer bytes (16x fewer than JSON), far more robust to bit corruption, and every English explanation parses back to its exact message. See [MLCP.md](MLCP.md).

## Talk to Mycelia (new)

`python -m mycelia.talk` for a command chat over real Mycelia data, or `--claude` for plain English through Claude, which can only answer by calling Mycelia's commands. See [TALK.md](TALK.md).

## Pure Mycelia (new, non-neural)

Rule-based agents learn invented-word meanings and compositional rules from demonstrations by sharing verified discoveries: 100% on unseen combinations with ~90 evaluations per episode, versus ~20,000-110,000 for a single joint-search learner. No neural networks or LLMs. See [PURE-MYCELIA.md](PURE-MYCELIA.md).

## Pure Mycelia v0.2: skills from online docs (new)

Reads docs.python.org through the allowlisted gateway, verifies each documented skill by experiment in a restricted sandbox, shares verified skills on the bus and composes them to solve held-out tasks, abstaining when it can't. Live run pending network access. See [PURE-WEB.md](PURE-WEB.md).
