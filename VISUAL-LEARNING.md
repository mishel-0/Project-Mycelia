# Label-free visual memory

This experimental path adds higher-resolution visual cues, local colony
specialization, and error-aware competition. It uses NumPy and Pillow, with no
CNN, neural network, backpropagation, tumor labels, or separately fitted readout.
It preserves the earlier labeled learner and its saved models.

## What learns

`VisualMycelium.encode_image(image)` now preprocesses an image with the exact
cue mode saved in its configuration; `learn_image(image)` encodes and learns it
in one call, while `recognize_image` and `reconstruct_image` reuse that encoding.
The lower-level `learn(cue)` still accepts a cue array. None of these methods
accepts categories, image IDs, patient information, or filenames. Image
preprocessing follows the mode recorded in the checkpoint and produces 32x32
intensity, edge-magnitude, and local-texture cues. The full-data CLI defaults
to fixed full-frame resizing; the legacy foreground-crop/contrast-normalization
path remains selectable. That crop is not anatomical skull stripping;
preprocessing settings are hypotheses to test.

The image is divided into sixteen spatial 8x8 patches. Colonies compete by actual
compartment receptor mismatch plus local cord mismatch. A sufficiently unfamiliar
patch recruits a new specialist while capacity and recruitment reserves permit.
Each scaffold's entire initial carbon, water, and energy comes from explicit
reserves; imprinting and subsequent cord chemistry also consume resources.
Recruitment models externally funded inoculation, not growth from a parent hypha.

Competition uses reconstruction errors. Familiar patches below a threshold skip
updates; other patches receive a bounded, error-dependent correction to the
winning colony. This does not correct tumor-class mistakes: none are available in
unlabeled training. Once the bank fills, unfamiliar patches share existing memory.
An experimental `motif_merge_threshold` can fuse two highly similar live
specialists at capacity, combining their receptor/route traces and local context
links before the freed slot is reused. It defaults to `0` (off): a `.03` threshold
appeared harmful and `.012` showed mixed held-out results in the earlier
pre-correction evaluator. Those comparisons are superseded pending a mask-safe
ablation; fusion is not a validated capacity fix.

The full-data training and held-out reconstruction CLIs now default to
multi-scale cues after a two-seed masked-transfer ablation: channel 0 keeps the
same grayscale values, while edge and texture channels combine native
gradients/5x5 texture with broader 13x13 statistics. On each seed, leakage-safe
held-out patch MSE improved by 6.8-7.3% and novelty overflow fell by 43-45%
under the then-current preprocessing protocol.
`--no-multiscale-features` restores the single-scale setting. Core
`VisualConfig` keeps the feature flag explicit so old callers and saved cues
are not silently reinterpreted.
A fresh 600/200 comparison on Testing (400 earlier Testing paths excluded)
again favored multi-scale cues: MSE fell from 0.024213 to 0.022925, with a
paired image-level 95% interval for the change of [-0.001689, -0.000904]. This
is still an image-level patch-prediction result, not a diagnosis or
patient-independent evaluation.

## Whole-image identity and episodic context

The image-level episodic bank stores a compressed, canonical uint16 visual cue
and a content fingerprint for each unique image representation, up to
`max_episodes` (10,000 by default). Exact duplicate cues share one identity.
No external filename, patient ID, or class label is accepted by this memory.
The `recognize(cue)` API returns an exact identity for a stored canonical cue;
for a non-exact input it reports only a nearest perceptual candidate and its
distance, or `novel`. Those distances are not probabilities.

Exact recognition of a seen image is closed-set episodic recall. It is expected
to reach 256/256 for 256 unique stored queries, but it does not show that the
colonies inferred meaning or can recognize unseen images. The recall CLI reports
identity separately from cue-addressed reconstruction. For context transfer,
`predict_masked_patch(cue, patch_index)` selects an episode using all other
patches and predicts the hidden spatial patch from that episode. The
`predict_masked_patch_from_colonies()` path uses learned local directional
associations between neighboring patch specialists and never consults the
whole-image episode bank. The
`tools/masked_visual_prediction_test.py` benchmark evaluates these paths on a disjoint,
unlabeled held-out set against a training spatial-mean baseline and a same-image
adjacent-patch baseline. By default it trains on unique `Training` images and
evaluates unique `Testing` images, excluding source-pixel and encoded-cue
duplicates across both sets. The dataset does not provide verified patient IDs,
so this remains an image-level test rather than a patient-level clinical test.
For each hidden target, that benchmark fills its grayscale patch using only the
visible-image mean before recalculating edge/texture cues, preventing local
receptive fields from leaking hidden target pixels into neighboring patch
assignments. Its default `--preprocessing fixed` path resizes the complete
image to a fixed canvas and uses no image-dependent cropping or contrast
statistics. The optional `--preprocessing foreground` path reproduces the
older transform and reports its global-statistic caveat. Earlier masked-patch
scores computed before masking are superseded.

Two fresh strict-mask runs trained on 600 unique unlabeled Training images and
tested 200 previously unevaluated Testing images apiece, with 3,200 masked
patches per run and zero source-pixel or encoded-cue overlap. Learned colony
context MSE was 0.01531 vs 0.02061 for the training spatial mean in one split,
and 0.01405 vs 0.01930 in the second (25.7% and 27.2% lower). It beat that
baseline on 87.0% and 83.5% of images, respectively; image-bootstrap 95%
intervals for context-minus-mean MSE were [-0.00662, -0.00413] and
[-0.00652, -0.00410]. This supports image-level grayscale context transfer
under fixed preprocessing. It does not demonstrate tumor semantics, patient
independence, diagnosis, or fungal equivalence. Details and raw records are in
the output report `STRICT-MASKED-TRANSFER-RESULTS.md`.

The full-data training CLI enables pair-conditioned context motifs by default,
with `--no-hierarchical-context` as an opt-out. The comparative masked-transfer
CLI keeps them off by default; pass `--hierarchical-context` for the matched
ablation. Each motif uses two direction-tagged neighboring colony identities
to predict the target colony, and backs off to first-order context when an
unseen pair has no stored association. In two fresh, disjoint 600/200 runs, the
hierarchy lowered MSE from 0.015763 to 0.013524 and from 0.015989 to 0.013674.
It beat first-order context on 84.5% and 90.5% of images, with paired image
bootstrap intervals for the change of [-0.002743, -0.001764] and
[-0.002836, -0.001783]. Motif-conditioned prediction covered 91.0% and 92.25%
of patches; remaining cases backed off to first-order routes. This supports
the hierarchy for label-free 2D grayscale context, not anatomy recognition or
medical diagnosis. See `HIERARCHICAL-CONTEXT-ABLATION.md` for the complete
protocol and raw records.

## What the tests mean

Local association weights learn from directional neighboring-colony surprise.
An optional mode also uses pre-learning masked-patch prediction error to
modulate local receptor updates. Same-split development results were nearly
identical with this mode on or off, and the separate Testing result was
slightly worse with it on, so it is disabled by default pending calibration.
Context-association learning remains enabled.

Recall is cue-addressed: patches in the input select stored graph imprints. The
output consists of stored receptor values assembled in the input's patch positions.
The model is not an unconditional image generator. It does not copy the query's
pixel values directly into the returned reconstruction.

- Reconstruction MSE/RMSE measures visual information retained in graph state.
- Training-only unseen-image probes measure reconstruction on images not fed to
  those models. Source and encoded duplicates are excluded from those splits.
- Final all-data training feeds every raw Training and Testing image. Its subsequent
  retention measurements are on seen images, not unseen medical accuracy.
- An external evaluator can match remembered reconstructions to original images.
  Source identity retrieval is not disease classification and is not part of the
  learner or its checkpoint. Exact repeated sources count as one identity.
- Episodic identity uses the model's canonical cue fingerprint and stored visual
  template. Report it as seen-image retrieval, separately from reconstruction-
  based source matching and all held-out results.
- Receptor erasure must remove supported recall, and recall must leave frozen
  graph state unchanged. Save/load checks and resource accounting guard integrity.

Patient identities are unavailable. No patient-independent medical result is
claimed. These memory rules remain computational hypotheses inspired by biology;
the dimensional growing `mycelia.biology.Hypha` engine is still a separate module.

In strict fixed-preprocessing confirmation runs, 600 unlabeled Training images
built each model and 200 fresh Testing images supplied 3,200 masked patches.
Colony-context RMSE was 0.1237 and 0.1185, compared with spatial-mean RMSEs
0.1436 and 0.1389. Source and encoded-cue overlaps were zero. This is
image-level context transfer, not semantic understanding or a clinical result.
At capacity, fully retired colony slots can now be reused; live colonies are
retained, so finite live-specialist capacity remains. Slot reuse recycles the
allocation in the model's accounting and is not a detailed biochemical recovery
simulation. See the output report for strict-mask metrics; historical run
folders are retained as provenance and may use superseded protocols.

## Run

```sh
python tools/unlabeled_visual_test.py \
  --dataset Brain-Tumor-MRI-Dataset --output runs/visual-learning
```

`--probe-only --fix-probe` compares correction thresholds and capacity using only
the Training split. `--skip-probes --max-colonies 256 --correction-threshold .004`
can perform a fresh all-data fit after choosing settings from those probes. The
report must identify which settings were actually selected and executed.

Whole-image lessons are transactional: failure restores touched colonies,
recruitment reserves, current specialist support, merge counters, and selection
counts. A bank containing only retired tissue
cannot learn at capacity. Corrupt checkpoints and inconsistent counters are rejected.
