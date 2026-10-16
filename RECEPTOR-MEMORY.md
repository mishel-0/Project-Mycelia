# Receptor-Field Memory: Unsupervised Receptors + Teach-Only Recall

A new learning method for Project Mycelia that raised held-out brain-tumor MRI
accuracy from **73.3% to 94.5%** without a CNN, without backpropagation, and
without reward/punishment.

> Research prototype on one public dataset. Not a medical device and not
> clinically validated. See *Limits* before quoting these numbers.

## Results (held-out Testing split, scored once)

Dataset: `Brain-Tumor-MRI-Dataset` (4 classes). After removing duplicates,
label conflicts and Training/Testing overlap: **5,410 Training, 1,467 Testing**.
Settings were chosen on a 20% validation split of Training only.

### Diagnosis (4-class)

| Method | Accuracy | 95% interval |
|---|---|---|
| Original colony memory (`train_mycelium.py`, 16px) | 73.3% (1075/1467) | 71.0–75.5% |
| Reward + punishment (`reward_punishment_training.py`) | 63.9% (937/1467) | 61.4–66.3% |
| **Receptor-field memory (this method)** | **94.5% (1386/1467)** | **93.2–95.5%** |

Validation accuracy of the selected setting: 97.3%. Balanced accuracy on Testing: 94.4%.

| Class | Images | Recall | Precision |
|---|---|---|---|
| glioma | 379 | 82.1% | 98.4% |
| meningioma | 298 | 95.6% | 91.4% |
| notumor | 390 | 100% | 90.9% |
| pituitary | 400 | 100% | 97.6% |

Confusion matrix (rows = truth, columns = prediction; glioma, meningioma, notumor, pituitary):

```
glioma      311  27  37   4
meningioma    5 285   2   6
notumor       0   0 390   0
pituitary     0   0   0 400
```

The main remaining error is glioma: 27 called meningioma and 37 called no tumor.

### Tumor presence (tumor vs. no tumor)

Derived from the same predictions (glioma, meningioma or pituitary = tumor).

| Metric | Value |
|---|---|
| Accuracy | **97.3%** (1428/1467) |
| Sensitivity (tumors detected) | 96.4% (1038/1077) |
| Specificity (healthy scans cleared) | 100% (390/390) |
| Positive predictive value | 100% |
| Negative predictive value | 90.9% |
| Missed tumors | 39 (37 of them glioma) |

## How it works

**Stage 1: unsupervised receptor growth (no labels).**
Images are resized to 64×64 grayscale. From each Training image, 20 random
6×6 patches are contrast-normalised and whitened. 800 "hyphal tips" compete for
these patches: each patch goes to the tip it matches best, and each tip then
grows toward the patches it won (competitive learning, i.e. spherical k-means).
The result is a dictionary of local receptors learned without any label.

**Stage 2: cord pooling.**
Every 6×6 window of an image is matched against all receptors (positive and
negative rectified responses), and responses are averaged over the whole
image, its 4 quadrants and a 4×4 grid, like cords collecting signal from regions
of the network. That gives 33,600 response values per image.

**Stage 3: teach-only memory.**
Each labelled Training image, plus its left-right mirror (brains are roughly
symmetric), is stored as a memory trace. A new image is compared to every trace
with an exponential similarity, and the trace weights come from one closed-form
linear solve (kernel ridge regression). There is no reward/punishment and no
gradient training.

### Mapping to basal-cognition ideas

| Biological idea | What implements it here |
|---|---|
| Tips compete for resources and specialise | Competitive receptor learning (Stage 1) |
| Cords thicken along useful paths | Regional pooling of receptor responses (Stage 2) |
| Structural memory of past encounters | Stored traces of taught images (Stage 3) |
| Bioelectrical spikes, chemical priming | **Not implemented yet** |

These mappings are analogies. The computation itself is standard unsupervised
feature learning plus kernel memory, and is described as such.

## What did not help

- **More epochs** of the original colony memory: it had plateaued around 73%.
- **Reward/punishment**: lower than teach-only (63.9% vs 73.3%). Punishment
  wore down shared glioma/meningioma features.
- **Higher resolution** (96px, 128px): no gain over 64px.
- **Small shifted imprints** in addition to mirrors: no validation gain.

## Limits

- **Near-duplicates.** Plain 1-nearest-image matching on raw 64px pixels already
  scores ~90% on Testing, so many Testing images closely resemble Training
  images. The score measures performance on this dataset, not on new patients.
- **No patient identity.** The split is image-level; slices of the same patient
  may sit on both sides.
- **Development choices were explored on Testing** before the validation
  protocol was fixed (resolution, receptor count). The final setting choice
  (mirror, gamma, lambda) used validation only.
- **Not clinical.** Missing 3.6% of tumors (mostly glioma) would be unacceptable
  for diagnosis.

## Reproduce

```bash
pip install -e .[images]
unzip Brain-Tumor-MRI-Dataset.zip -d data
PYTHONPATH=.:tools python tools/receptor_memory_mri.py \
  --dataset data/Brain-Tumor-MRI-Dataset --output results/receptor-memory
```

Runs in about 30 minutes on 4 CPU cores with 16 GB RAM, and writes
`results/receptor-memory/receptor-memory-results.json`.

Code: `mycelia/receptor_memory.py` (method), `tools/receptor_memory_mri.py`
(experiment), `tests/test_receptor_memory.py`.

## Image reconstruction

`ReceptorField.sense(img)` records every 6×6 window's receptor responses plus
its brightness and contrast; `ReceptorField.reconstruct(sensed, keep=None)`
rebuilds the image.

- **All receptors (default): lossless.** The 800 receptors span every 6×6
  patch shape, so the rebuild is exact. Full-resolution Testing MRIs
  (225–512 px) come back with **0 gray-level error**.
- `keep=k`: only the k strongest receptors per window (lossy compression):
  10 receptors gave 38–48 dB at full resolution; k ≥ 36 is exact.

Audit of the first attempt (`research/.../recon.py`): it was lossy because it
kept only 1–10 receptors per 36-pixel window and compared against a 64×64
downscale, not the original. The classifier's pooled memory cannot rebuild
images: regional averaging discards position inside each region.
Script: `research/receptor-memory-exploration/recon_fullres.py`.

## Real mycelial network topology (`mycelia/mycelial_network.py`)

The receptor field now runs inside an actual `Mycelium` lattice (6×6
compartments, 60 septal cords, built like `MemoryColony`). Each compartment
senses one image region. Cords are thickened by unsupervised Hebbian learning
(compartments whose responses co-vary across unlabelled Training images),
paid from compartment nutrient and ATP-like energy; budgets validate exactly
(carbon and energy error 0). Compartment states then spread along cords
weighted by real conductance (∝ radius⁴) before teach-only memory.

Held-out Testing (`tools/mycelial_network_mri.py`, 1,467 images):

| Variant | Diagnosis | Tumor presence | Sensitivity | Specificity |
|---|---|---|---|---|
| Isolated compartments (validation-selected) | **94.8%** (1391) | 97.2% | 96.3% | 99.7% |
| Spread on untrained cords | 94.8% (1390) | 97.6% | 96.7% | 100% |
| Spread on learned cords (radius 0.315→0.41) | 94.5% (1387) | 97.4% | 96.6% | 99.7% |

Finding: the compartment lattice gives a small gain over the pooled version
(94.5%), but signal spread along cords, learned or not, does not improve
diagnosis; differences are within noise. Spreading blurs tumor location,
which is what separates the classes. Validation tied isolated and learned
cords (97.6%); ties go to the first variant. Glioma recall remains the weak
point (82.8%).
