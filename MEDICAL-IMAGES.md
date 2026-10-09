# Dimensional-core medical-image experiment

This experiment tests the new `mycelia.biology.Hypha`, not the older graph prototype or engineered receptor-memory module. It has two distinct outputs:

1. A label-free physiological response and numerical-integrity test on all available images.
2. An explicitly external, Training-only numerical readout measuring how much class-label information those responses retain on held-out images.

The new biological core is not trained and has no image-learning law. No CNN, neural network, SVM, sklearn, learned image encoder or backpropagation is used. The external readout is NumPy standardization plus Euclidean distance to four Training-class centroids. It is a measurement instrument for this experiment, not a fungal learning mechanism.

## Reproduce

```bash
python -m pip install -e '.[dev,images]'
python tools/medical_image_test.py --dataset /path/to/Brain-Tumor-MRI-Dataset --output runs/medical-images
```

After the test, predict a single image with the same external readout:

```bash
python tools/predict_medical_image.py /path/to/image.jpg --readout runs/medical-images/external-physiology-readout.json --output runs/single-image
```

This command validates the saved protocol and core-source fingerprint, then produces an experimental dataset label and fresh physiological state. It does not update either the readout or the core. Predictions do not represent clinical diagnoses. The feature cache likewise verifies decoded image hashes, the fixed protocol, parameters and core source fingerprint before reuse.

Optional scientific plots use Matplotlib:

```bash
python -m pip install -e '.[plots]'
python tools/plot_medical_image_results.py runs/medical-images
```

The dataset must have `Training/<class>/*.jpg` and `Testing/<class>/*.jpg`, with classes `glioma`, `meningioma`, `notumor`, `pituitary`. Labels are read from folders for split analysis and the external readout; they never enter `encode_image()` or the physiology equations. Corrupt inputs or failed simulations abort fitting instead of silently removing difficult examples.

## Fixed synthetic input protocol

Each MRI image is decoded to RGB, hashed at its original dimensions, converted to Pillow grayscale and BOX resized to 8×8. The intensity scale is always gray/255, without per-image min-max normalization. The row-major scan groups adjacent intensities into 32 pairs. One pair drives the basal and apical baths for 0.1 min at nutrient concentration `0.02 × intensity` M. Each bath receives a **finite amount** equal to concentration times its current water volume. The core's new `set_bath_nutrient()` intervention records both externally added carbon and removed residual substrate.

Each image starts a fresh two-compartment hypha with identical water, initial turgor, soluble resources, wall and reserve pools. Initial ATP and vesicle cargo are zero. Image intensities do not initialize hidden seed state. Local physiological reactions remain unchanged. Total simulated exposure is 3.2 min with a maximum internal substep of 0.01 min.

The response vector contains eight internal observables after every pulse (256 values): tip turgor, internal osmotic pressure, ATP equivalents, apical wall cargo, permanent extension, pressure-driven flow toward the apex, tip soluble nutrient and total intracellular water. External bath concentrations, labels and budget ledgers are not readout features. Thus an intensity can affect the readout only through evolving internal state. This synthetic scan is a sensor encoding, not a spatial MRI-grown fungal colony or measured MRI-to-chemistry relationship.

## Independent readout evaluation

Exact duplicates use SHA256 of original decoded RGB dimensions and bytes. Exclude conflicting-label hashes, within-Training duplicates, Testing hashes present in Training, and within-Testing duplicates from accuracy evaluation. Also deduplicate the actual 8×8 uint8 input stimulus, excluding encoded train/test overlaps, within-split encoded duplicates and encoded conflicting labels. Physiological testing still processes every raw image. Report source-pixel counts and the stricter stimulus counts separately.

All readout normalization and class centroids are fitted on Training only and frozen before Testing. The protocol and classifier type are fixed before inspecting results; there is no Testing parameter search. Compare:

- physiology response only;
- 64 direct grayscale pixel features;
- direct pixels plus physiology;
- global mean intensity only;
- Training-majority baseline;
- physiology centroids fitted with one seeded permutation of Training labels.

Report overall accuracy, balanced accuracy, confusion matrices, per-class recall/precision/F1, image-level Wilson accuracy intervals, and paired bootstrap accuracy differences versus direct pixels. A single label permutation is a negative control, not a permutation significance test. A confidence interval crossing zero does not establish that physiology improves classification.

## Numerical and input controls

Use the first two clean Testing images per class, chosen by file order rather than prediction correctness:

- repeat and feature-cache recomputation;
- restart halfway through the fixed scan;
- histogram-preserving pixel permutation;
- a uniform image with the same mean intensity;
- disabled input-sensitive physiology, which should remove image-dependent response;
- refined 0.005-min internal timestep, compared using fixed characteristic scales recorded in `biology/images.py`.

Carbon, water, ATP-equivalent and bath-salt ledgers are checked after every internal step. Report maximum recorded residual and numerical water-limiter activation. These controls test causality and computational behavior, not clinical usefulness or biological recognition.

## Outputs and limits

`report.html`, `RESULTS.md`, `summary.json`, per-image physiology/prediction CSVs, feature cache, external frozen readout JSON, and four examples with original inputs, physiology traces and restart states are produced. The same data may contain unknown patient overlap, acquisition differences and near duplicates. Patient identifiers and provenance are unavailable in the supplied folder dataset. Exact-pixel filtering does not remove these risks.

This is an exploratory **image-level** discrimination test. Intervals assume independent images. No clinical accuracy, diagnostic benefit, biological fidelity or fungal image-learning claim follows from these results. Existing prototype image-memory accuracy is a separate experiment and must not be presented as a result for the new core.
