# MYCELIA medical-image architecture: measured gap and next gates

This document turns “understand a medical image” into separate tests. A single
accuracy number cannot establish reconstruction, diagnosis, efficiency, data
efficiency, or fungal resemblance. OMNIA is the lossless input/container layer;
MYCELIA is the learner.

## Current measured state

| Requirement | Current evidence | Status |
| --- | --- | --- |
| OMNIA source preservation | One full-resolution JPEG payload per container; 7,200-file conversion/CRC/pixel audit passed | Implemented as storage |
| Unlabeled image learning | All 5,410 clean Training images reached `VisualMycelium.learn`; labels were excluded | Implemented |
| Four-class image prediction | 980/1,468 = 66.76%; balanced accuracy 65.78% | Below target |
| Conventional non-neural comparator | HOG/intensity RBF-SVM: 1,315/1,468 = 89.58%; balanced accuracy 89.19% | MYCELIA is behind |
| Few-label prediction | With a representation already trained on all 5,410 unlabeled images: 200 labels gave 62.23% mean balanced accuracy; 2,000 labels gave 65.66% | Few-label advantage not demonstrated |
| Exact full-source recreation by MYCELIA | 256-image audit: 32px preprocessing ceiling averaged 21.58 dB PSNR; learned colony reconstruction averaged 18.92 dB | Architecturally impossible today |
| Efficient inference | Compiled frozen graph preserves every score/prediction; 3.09 MB and 0.014 s load versus 158.04 MB and 2.07 s | Improved |
| Efficient training | Final MYCELIA fit: 21.07 min. Conventional comparator final fit: 1.28 s, excluding its 0.31 s descriptor extraction | MYCELIA is behind |
| Patient-independent testing | Current 2D dataset has no verified patient identifiers | Missing |
| Biological resemblance | Resource accounting and graph traces exist; active image colonies are fixed lattices and are uncalibrated against fungal time series | Not demonstrated |

The conventional comparator is retrospective because this Testing set had
already been opened. It is a diagnostic benchmark, not a new confirmatory
claim. Its large advantage shows that oriented boundaries and spatial layout
are much more useful than the current colony summary. More colony capacity by
itself will not close this gap.

## What “understanding” must mean in this project

MYCELIA must pass each gate independently:

1. **Source fidelity:** OMNIA decodes to the exact source pixels and metadata.
   This tests the container, not learning.
2. **Perceptual reconstruction:** from a masked or corrupted unseen scan,
   predict withheld voxels and tumor-region structure. Report PSNR/SSIM and
   errors inside and outside expert masks. Copying an input payload does not
   pass this gate.
3. **Anatomical representation:** keep spatial position, MRI sequence, voxel
   spacing, orientation, and 3D patient study together. The current 32×32
   single-image cue cannot pass this gate.
4. **Diagnostic discrimination:** predict a defined reference label on
   patient-separated internal testing and an untouched external hospital.
   Report sensitivity, specificity, balanced accuracy, calibration, and
   abstention for unfamiliar scans.
5. **Data efficiency:** compare learning curves while varying unlabeled
   patients and labeled patients separately. A model trained on thousands of
   unlabeled images is not four-shot learning merely because its readout sees
   four labels.
6. **Compute efficiency:** on the same machine and cohort, measure total data
   conversion, representation training, readout fitting, peak memory, artifact
   size, energy if instrumented, and end-to-end latency. Compare with fixed
   non-neural and neural baselines; do not compare selected stages only.
7. **Biological behavior:** match measured fungal growth trajectories under
   nutrient, confinement, obstacles, grazers, and damage. Image accuracy does
   not count as biological validation.

## Required architecture

### 1. OMNIA study format

Retain the original image/volume payload losslessly and add a manifest that
groups patient → study → sequence → volume. Preserve affine geometry, voxel
spacing, orientation, acquisition sequence, and expert masks when available.
The learned representation and any residual/payload must be separate fields so
“reconstructed from memory” cannot silently read the source bytes.

### 2. Streaming multi-resolution sensory field

Replace the single 32×32 view with a streamed pyramid. A practical first gate
is 64×64 for 2D debugging, followed by aligned 3D 4×4×4 voxel patches. Retain
intensity, signed oriented gradients, local texture, physical position, and MRI
sequence. Stream bounded batches from OMNIA instead of holding every cue in
RAM. Version the feature schema; old checkpoints remain readable only by their
old schema.

### 3. Locally predictive colonies

Each colony receives a local patch plus physical position and sequence. It
predicts masked neighbors and cross-sequence patches. Only local prediction
error changes its receptor state and nearby routes. Classification labels never
enter this stage. A reconstruction head must predict all retained low-frequency
3D modes first, then progressively higher-frequency residuals. Source-exact
recreation requires a lossless residual in OMNIA; learned generalization is
measured without that residual.

### 4. Grown sparse topology

The active image architecture must replace prebuilt 8×8 lattices with tips that
branch toward persistent prediction error, fuse compatible routes, reinforce
useful transport, and prune/recycle routes with poor benefit per resource cost.
Growth choices must be local and transactionally resource-funded. A fixed node
cap needs living-slot competition; the current run recorded 6,546 novelty
overflows and no recycling or motif fusion.

Topology is evaluated with route length, transport cost, redundancy, recovery
after controlled edge removal, branching-angle/velocity distributions, and
time-series similarity to organisms. These biological scores remain separate
from MRI scores.

### 5. Diagnosis and localization readouts

After unlabeled learning is frozen, fit a transparent supervised readout for:

- patient/study-level class prediction;
- voxel/region localization against expert masks;
- uncertainty calibration and abstention;
- out-of-distribution detection for new scanners, sequences, and pathologies.

Nearest centroid remains a diagnostic baseline. Linear/RBF readouts may test
whether information exists in the representation, but their contribution and
training cost must be reported separately. The architecture passes only if the
MYCELIA representation improves matched readouts over raw handcrafted inputs.

## Data already available locally

- `RHUH-GBM-derived-28patients`: 28 subjects, 84 visits, four MRI sequences,
  336 volumes and 336 masks. This supports patient-separated 3D reconstruction
  and tumor-region tests, but it is GBM-only and cannot validate four tumor
  classes or healthy specificity.
- `work/rhuh-gbm`: 40 subjects, 120 visits, 720 NIfTI files. It needs provenance,
  checksum, sequence and alignment reconciliation before combination with the
  28-patient derivative.
- `brain_tiff`: 200 slices from 10 patient prefixes. It supports a small grouped
  2.5D pilot; filenames do not supply physical 3D geometry.
- The current 7,200-image OMNIA collection has four image labels but no trusted
  patient IDs. It remains useful for engineering, not patient-independent proof.

To support a >95% claim, acquire a patient-identified multiclass cohort and a
different-hospital external cohort with compatible reference labels. No image
count can guarantee 95%; power and confidence depend on class prevalence and
the desired confidence-interval width. For example, even 95% observed accuracy
on 1,000 independent patients has an approximate binomial standard error near
0.69 percentage points before subgroup analysis.

## Compute plan

The current 32px, 1,024-colony final fit took 1,264 s for 5,410 images. Simple
area scaling estimates about 84 min at 64px, 5.6 h at 128px, and 22.5 h at
256px if patch size and colony count stay fixed. These are planning estimates,
not measured runtimes; the current patch-update loop can scale worse.

Before higher-resolution training:

1. use the compiled numeric scoring arrays during frozen inference;
2. replace global per-image rollback copies with per-key journals while keeping
   failure rollback tests;
3. stream cues and features from checksum-bound caches;
4. benchmark 64px on a Training-only subset and measure peak memory;
5. stop configurations that fail a predefined reconstruction/representation
   gate before a full fit.

The compiled matcher already reduces the deployment checkpoint from 158.04 MB
to 3.09 MB, cold load from 2.07 s to 0.014 s, and warm single-image colony
features from 53.4 ms to 0.87 ms on the measured Apple M5 system, with bitwise
feature parity and identical class scores on all 1,468 saved predictions. It
does not reduce the original learning time or raise accuracy.

## Biological calibration datasets

Use true fungi and preserve species/condition identity:

- The open [Fungus Olympics study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0257823)
  includes six maze time-lapse videos plus source measurements across honeycomb,
  square, and boomerang obstacles. It reports velocity and branching behavior
  for multiple filamentous fungi.
- The [Dryad confinement dataset](https://datadryad.org/dataset/doi%3A10.5061/dryad.76hdr7ss1)
  provides four time-lapse videos (45.23 MB total) and measured confined growth
  velocities for *Talaromyces helicus* and *Neurospora crassa* under controlled
  nutrient/water conditions.
- The [Fungal Network Grazer dataset](https://zenodo.org/records/14931953)
  provides original time-series images and processed networks for four fungal
  species with and without grazers. It is 10.2 GB, so begin with one species
  only after the video pipeline passes the smaller benchmarks.

The first admission test should use the small maze and confinement videos:
extract tip position, velocity, branch events, turn choices, and response to
walls; compare simulated distributions and trajectories before adding more
biology. Growth laws that fail these tests should not be described as
fungal-equivalent.

## Next executable phase

1. Build a 64px orientation-aware local cue schema and a sparse spatial
   readout. Select it on Training validation only. Gate: exceed the 89.58%
   conventional comparator on Training validation before opening any fresh
   test cohort.
2. Run an unlabeled-data curve as well as the completed label curve. Gate:
   demonstrate improvement over raw cues at low unlabeled and labeled counts.
3. Move the successful mechanisms into the patient-grouped 3D RHUH pipeline.
   Gate: masked reconstruction and tumor-mask alignment improve over matched
   raw/DCT/ridge controls across patient folds.
4. Implement grown branching/fusion/pruning in the active visual topology and
   validate it on fungal videos. Gate: predeclared morphology/response metrics,
   not visual resemblance alone.
5. Freeze architecture, obtain a fresh patient-level external cohort, and test
   once. The >95% goal is accepted only if the lower confidence bound and
   clinically relevant per-class sensitivities meet the predefined threshold.

### First 64px gate result

The first Training-only 300/400-image ablation is complete. A spatial pyramid
over 64px edge/texture colonies improved fixed-SVM balanced accuracy from
71.75% at 32px to 76.75%; combining 32px and 64px reached 77.25%. Signed
orientation, axis-specific edge energy, multi-receptor fusion, and doubling
capacity to 512 did not beat that result. The matched HOG/intensity control was
80.0%. See [ORIENTATION-ABLATION.md](ORIENTATION-ABLATION.md). The evidence
rejects “add more colonies” and “add signed gradients” as the next step. The
next gate is a phase-tolerant local frequency/shape representation with true
masked-region prediction.

No existing evidence supports a claim of cancer diagnosis, 100% learned source
recreation, superiority to traditional AI, or full biological equivalence.
Those remain project goals with explicit gates above.
