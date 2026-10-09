# Full-data graph memory training (MYCELIA 1.2.0)

This workflow trains `AssociativeMycelium`: persistent, carbon- and energy-funded receptor traces in actual prototype compartments, with local cord reinforcement. The newer dimensional `mycelia.biology.Hypha` physiology core remains a separate module and is not trained here. The equations and image-to-chemical interface are computational hypotheses, not established fungal learning laws.

## Reproduce

```bash
python -m pip install -e '.[images,dev]'
python -m pytest -q
python tools/train_mycelium.py --dataset /path/to/Brain-Tumor-MRI-Dataset --output runs/full-data-memory --epochs 2 --seed 42
python tools/predict_memory_image.py --memory runs/full-data-memory/full-data-memory.json --image /path/to/new-image.jpg
```

Only NumPy and Pillow are used for training and inference. No CNN, neural network, backpropagation, SVM, learned image encoder or independently trained readout is used. `.[learning]` is unnecessary for this workflow. The mathematical algorithm is online prototype learning implemented in resource-funded graph state; physiological bookkeeping alone does not establish biological fidelity.

The dataset has `Training` and `Testing` folders, each containing `glioma`, `meningioma`, `notumor` and `pituitary` image folders. Archives omit the original dataset. Model checkpoints contain full node/cord state, not source image files. Supply your own dataset path to reproduce.

## Completed brain MRI run

The 7,200-image run selected **32 consolidated colonies per class** on Training-only validation (78.29% balanced validation accuracy). After refitting on 5,410 clean Training images, frozen unseen-image accuracy was **76.89% (1,128/1,467)**, with a 95% Wilson interval of 74.67%–78.98%. Held-out recall was 65.44% for glioma, 59.73% for meningioma, 89.74% for notumor and 88.00% for pituitary. The majority baseline was 25.84%.

The final separate all-data model completed **14,400 lessons**: two passes over every original image. Its training recall was **81.21% (5,847/7,200)**. The bounded population compresses exposures into imprints; it does not retain every image perfectly. This recall result does not estimate future-image accuracy.

Erasing receptor traces caused all 1,467 held-out predictions to abstain. Cord ablation retained 76.89% accuracy, with some individual predictions changed: this run shows no overall accuracy benefit from learned cord traces. Both checkpoint round trips preserved predictions, frozen inference preserved state, and the maximum carbon/water/energy residual remained below 1e-8. CLI predictions agreed with the batch output in a minimal runtime without sklearn, PyTorch or TensorFlow. The suite passed 108 tests in the project environment; the minimal runtime passed 106 and skipped the optional SVM test module.

## Evaluation and final model

1. Decode every RGB image, hash decoded pixels and the actual 16x16 grayscale input, and reject unknown classes or invalid images. Quarantine conflicting labels; a conflicting dataset cannot proceed to final all-data fitting.
2. Remove repeated source or encoded inputs within each split. Remove all Testing overlap with every original Training source/encoded input. This protects against an image alias that happens to be removed from Training deduplication.
3. Split clean Training into stratified 80% inner-training and 20% validation, seed 42. Use up to 500 inner-training images per class for configuration selection. Train each candidate for two shuffled passes: 8 colonies/class with fixed rate, 16 colonies/class with consolidation, and 32 colonies/class with consolidation. Select by validation balanced accuracy, breaking ties by smaller capacity. Testing labels never select the configuration. This compact selection pool need not rank configurations exactly as full Training would.
4. Initialize a fresh winner and train on **all clean Training images** for two shuffled passes. Freeze it, save `heldout-memory.json`, and evaluate clean Testing once. `heldout-results.json` and `heldout-predictions.csv` preserve this result before fitting all data.
5. Initialize another fresh winner and train on **every valid original image in both Training and Testing**, including duplicate rows, for two passes. Save `full-data-memory.json`. These predictions in `full-data-recall-predictions.csv` are **resubstitution/training recall**, not independent accuracy. Capacity is bounded, so processing every image does not guarantee exact memorization of every image.

The encoder is fixed grayscale BOX resize plus two signed local gradients. Labels select a rewarded colony population. The first impressions initialize population imprints, then a lesson updates the closest existing colony within its label. Optional consolidation changes the subsequent receptor rate to `min(0.12, 1/(previous_local_exposures+1))`. Local resource limits still cap synthesis. Inference reads minimum compartment/cord mismatch across each label's population; lower values fit better and are not probabilities. The lattice scaffold is seeded, not grown from the MRI.

## Checks and corrections

The workflow tests frozen state hashes before/after inference, scalar/batch score equivalence, exact predictions after save/load, carbon/water/energy budgets, total training exposure counts, untrained abstention, receptor-erasure abstention and cord ablation on identical held-out images. `summary.json` stores configuration, metrics, Wilson binomial intervals, confusion matrices, class recall, learning evidence, checkpoint hashes and integrity checks. Intervals describe this image sample and do not correct for unknown patient correlation.

Version 1.2.0 charges adjoining compartments' energy for cord reinforcement, commits a new colony only after a successful paid lesson, makes turnover transactional, validates memory checkpoint configuration and scaffold registration, abstains on unfunded imprints and resolves version metadata disagreement. Batch recall builds temporary arrays directly from current graph state; no separate fitted or persistent cached predictor exists.

Exact duplicate filtering cannot establish patient independence, remove all rotated/cropped/contrast-adjusted duplicates, or provide external-site validation. The source folders lack reliable patient IDs. Model output is experimental image-level classification, not a diagnosis. Proposed molecular memory, ions/calcium, electrical signaling and species-specific kinetics remain unvalidated.

## Optional scientific report

```bash
python -m pip install -e '.[plots]'
python tools/plot_memory_training.py runs/full-data-memory --dataset /path/to/Brain-Tumor-MRI-Dataset
```

Open `report.html`. It links both saved models, metric JSON and all predictions. Exportable PNG/SVG figures and deterministic examples of correct and incorrect held-out predictions are included. The plotting extra is not needed to run a saved predictor.
