# OMNIA tumor-label research experiment

MYCELIA learned from all **5,410 clean Training images** without diagnosis
labels. A separate standardized nearest-centroid readout, fit afterward using
Training labels, correctly predicted **980 of 1,468 held-out Testing images
(66.76%)**. Balanced accuracy was **65.78%**; the image-level Wilson 95%
accuracy interval was **64.31–69.12%**.

The identical three-channel input-cue centroid baseline scored **63.22%**,
the grayscale-only centroid **62.47%**, and the Training-majority baseline
**25.82%**. The learned features improved over the matched-cue baseline by
**3.54 percentage points**, or 52 additional correct images in this run.
The earlier 76.89% supervised MYCELIA result uses a different model/protocol
and remains a historical reference, not this experiment's score.

## Protocol

1. Decode one full-resolution `.omnia` container per image through the user's
   OMNIA SDK. Labels remain in `manifest.csv`, outside the learner.
2. Use the saved fixed full-frame preprocessing: grayscale 32×32, with
   deterministic intensity, edge and texture cues including multiscale detail.
   OMNIA preserves full-resolution source data; the current learner consumes
   these reduced cues, not every full-resolution pixel.
3. Remove exact decoded-source and actual input-cue duplicates. Remove any
   Testing overlap with Training without consulting Testing labels. This
   removed 190 Training duplicates, 114 Testing overlaps, and 18 additional
   Testing duplicates from 7,200 input files.
4. Split clean Training into 4,328 inner Training and 1,082 validation images,
   stratified by Training labels, seed 42. Only cue arrays enter
   `VisualMycelium.learn()`. Fit a fresh unlabeled model for each colony cap.
5. Extract four spatial winner-colony histograms plus the three smallest
   colony mismatches for each of sixteen image patches. Fit the separate
   readout using inner Training labels. Select by validation balanced accuracy;
   ties favor the smaller cap.
6. Refit the selected unlabeled model from scratch on all clean Training,
   then fit the separate readout on Training labels and save both. Score
   Testing using the frozen artifacts, with no further model selection.

| Colony cap | Validation balanced accuracy |
| ---: | ---: |
| 128 | 65.40% |
| 256 | 67.39% |
| 512 | 69.99% |
| 1,024 | **70.93%** |

The final feature vector has 4,144 entries. The final learner completed
86,560 patch exposures and 83,970 local updates in one pass. It recorded
2,590 suppressed updates and 6,546 novelty-overflow events after capacity was
reached. Its new-colony funding pool was exhausted; existing colonies continued
updating. The maximum numerical resource-budget discrepancy was about
1.06×10⁻¹⁰ in the model's accounting units. These are computational resource
ledgers, not calibrated fungal measurements.

The original full run took approximately 65 minutes, including capacity
selection; its final fit took about 21 minutes on this machine. The identical
three-channel cue baseline was added afterward as a fixed, untuned control;
it used the same clean Training and Testing cohorts and did not change the
saved MYCELIA model, readout, or predictions. Future runs include that baseline
automatically. `matched-cue-control.json` preserves this provenance for this run.

## Run

From the source checkout, install the image dependencies and make the user's
SDK available. Use the SDK checkout and converted dataset that were already
verified; the SDK and dataset are separate from the model package.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,images]'
python -m pip install -e /path/to/omnia-sdk
python tools/omnia_tumor_research.py \
  --dataset-root /path/to/mycelia-omnia-brain-mri \
  --omnia-sdk /path/to/omnia-sdk \
  --output runs/omnia-tumor-research --seed 42
```

The dataset root must contain `manifest.csv` and `dataset/Training/*.omnia`
and `dataset/Testing/*.omnia`. The manifest needs `split`, `label`, and `omnia`
columns, where `omnia` is relative to the dataset root. All four class names
must be present in Training: `glioma`, `meningioma`, `notumor`, `pituitary`.
Do not substitute the old tiled 64×64 OMNIA dataset.

Predict from the saved artifacts:

```bash
python tools/predict_omnia_tumor_research.py /path/to/scan.omnia \
  --model-dir runs/omnia-tumor-research \
  --omnia-sdk /path/to/omnia-sdk --output prediction.json
```

The command checks the memory/readout pairing by SHA-256 and reloads the
model's preprocessing settings. It reports the predicted dataset label,
four standardized squared distances (smaller is closer), and the gap between
the smallest two distances. These values are research scores, not calibrated
probabilities or a percentage of image understanding.

## Evidence and limits

`summary.json` contains accuracy, balanced accuracy, per-class precision,
recall and F1, the confusion matrix, Wilson interval, learning/resource
counters, and integrity checks. `heldout-predictions.csv` records every held-out
prediction and its class distances. `capacity-selection.json` and
`input-audit.json` record model selection and the duplicate audit.

The full suite passed 267 tests. A native SDK miniature end-to-end run also
passed, including the added matched-cue baseline. Saved-model reload preserved
features and predictions; the actual single-image CLI matched the original
batch labels and distances for one predetermined image from each class,
including an incorrect prediction.

This is evidence of image-label information in MYCELIA's learned patch
representation. It does not establish full image understanding, localization,
clinical diagnosis, or biological equivalence to fungi. The readout uses
labels; the preceding visual learning does not. There is no CNN or neural
network in this learning/readout path.

Verified patient identifiers and tumor masks are unavailable. Exact duplicate
filtering does not remove every related slice or near duplicate. This result
and its confidence interval are image-level, not patient-independent or
external-hospital validation. The Testing set has now been evaluated; further
architecture tuning should use Training validation, followed by a fresh
independent test set for a new confirmatory result.
