# MYCELIA visual learner: architecture debug

Date: 2026-10-04. Scope: `mycelia.visual_memory.VisualMycelium`, its image
preprocessing and readout, plus the latest full-data identity and held-out
masked-patch runs. This is an engineering diagnosis, not a clinical validation.

## Evaluation correction

The earlier masked-patch benchmarks computed all image cues before hiding the
target patch. Because edge/texture windows overlap patch borders, neighboring
patches could retain some target pixels. Those masked-prediction scores
(including the prior 600/200 result) are superseded and must not be cited as
leakage-safe transfer evidence. The evaluator now replaces each target region
with the mean of visible pixels before recalculating cues and colony matches.
Source/cue duplicate filtering and label-blind learning remain in place.
The default benchmark now uses fixed full-frame resizing without image-
dependent crop or contrast statistics, then masks each target before feature
extraction. The older foreground crop/normalization path remains selectable
but reports its global-statistic caveat. Fresh strict-mask results are recorded
below; earlier masked-patch scores remain superseded.

## Finding

The central limitation is architectural, not a tumor-label problem in the
dataset. The current path can store an image and can memorize local patches, but
it has no mechanism that requires a representation to capture transferable
visual meaning. On one image, the colony state reconstructed the grayscale cue
with RMSE effectively zero. On a randomized 48-image probe, the 256-colony bank
filled by image 20. Later images still received exact IDs from the separate
episode store, while patch novelty overflow and reconstruction error rose.

An earlier 600/200 held-out experiment reported that nearest-episode context
did not transfer; its target patches were not hidden before feature extraction,
so those masked-patch metrics are superseded by the evaluation correction below.

## Root causes

1. **The original objective rewarded local storage, not understanding.** The
   visual learner now has a first local predictive rule: adjacent patch
   specialists form directional associations, reinforced in proportion to
   surprise. This learns local context without labels or episode lookup. It is
   first-order context can now back off through learned three-patch motifs. A
   strict-mask held-out test shows image-level transfer against spatial-mean
   and first-order baselines; it does not establish medical semantics.
2. **The representation is severely compressed and patch-independent.** The
   preprocessing converts every scan to 32x32 intensity plus fixed edge and
   texture channels. The learner then treats each 8x8 patch largely as its own
   matching problem. It has no 3D volume, MRI sequence, or anatomical
   registration, and its learned cross-patch relations are still limited to
   local neighbor associations. Fine tumor boundaries and subtle context can
   disappear in this bottleneck.
3. **The specialist bank saturates quickly.** With `max_colonies=256`, the
   random 48-image probe recruited 228 colonies by image 16 and hit 256 by
   image 20. At that point 411/768 patches (53.5%) had novelty above threshold
   without a new specialist. The 600-image fit reported 7,515/9,600 (78.3%)
   novelty overflow. When full, it continues adapting the nearest existing
   colony, which can blend unrelated patterns and cause interference.
4. **Retired-slot reuse was missing and is now implemented.** At capacity,
   `_recruit()` can rebuild a slot only when every node in that colony is
   retired. It preserves lifetime exposure accounting, reuses the slot's
   original resource allocation, and clears associations to the retired
   specialist. Live colonies are never evicted. A forced all-retired-bank test
   now successfully learns a new image and passes resource validation; an
   injected failure after replacement confirms full rollback. This fixes the
   dead-slot crash while leaving the live-bank capacity limit in place. The
   current accounting reuses a slot's funded allocation; it does not yet
   simulate biochemical recovery of material from dead tissue.
5. **Exact identity is supplied by a second, literal image store.** Each unique
   canonical 32x32x3 cue is compressed into an episode template and indexed by
   SHA-256. Exact recognition is therefore exact cue lookup, not semantic
   generalization. The 7,200-file run stored 6,878 unique cues and got 256/256
   exact sampled lookups; reconstruction-based source retrieval was only
   109/256 (42.6%). Those results answer different questions.
6. **“Adaptive” has only one active time scale here.** `_memory_config()` sets
   `adaptive=True` but explicitly sets `fast_rate=0`, disabling fast-trace
   plasticity. The visual learner therefore lacks the intended fast/slow
   adaptation in this configuration. Its slow trace update remains a local
   exponential adjustment, not predictive error correction.
7. **The visual topology is a fixed scaffold.** Each colony starts as a fully
   wired 8x8 grid. This learner does not branch image-processing hyphae, fuse
   useful physical routes, or grow a spatial network around a scan. The new
   predictive associations are a separate sparse graph between colonies, not
   branching/fusing fungal hyphae. `mycelia.biology.Hypha` is not the image
learner’s active topology. The full-data training CLI now enables the validated
pair-conditioned context layer by default; it remains a predictive association
graph, not a living or branching hyphal network.

## Diagnostic evidence

The first cue in the capacity probe used 16 new colonies and reconstructed at
RMSE 6.2e-18 from colony traces alone. At image 16, it had 228 colonies and
RMSE 0.0304. At image 20, it hit 256, recorded 15 novel-overflow patches in
that image, and RMSE was 0.1144. By image 32 the RMSE was 0.2363, although its
episode lookup remained exact. This is direct evidence of successful one-shot
local imprinting followed by a capacity/generalization failure—not evidence of
one-shot semantic learning.

Two fresh strict-mask runs trained on 600 unique unlabeled Training images and
tested two disjoint groups of 200 previously unevaluated Testing images. Each
scored all 3,200 patches after masking in fixed full-frame grayscale space;
source-pixel and encoded-cue overlaps were zero. Colony-context MSE was
0.01531 vs 0.02061 for a training spatial-mean baseline in run one, and
0.01405 vs 0.01930 in run two. Context beat that baseline on 87.0% and 83.5%
of images. Paired image-bootstrap 95% intervals for context-minus-mean MSE
were [-0.00662, -0.00413] and [-0.00652, -0.00410]. This proves transfer of
low-level grayscale patch context on these image-level splits—not tumor
recognition, patient-independent generalization, diagnosis, or biological
equivalence. Patient IDs remain unavailable. The current code writes schema-7
checkpoints with opt-in motif-fusion accounting and strict-mask held-out
evaluation. The output report contains raw per-patch rows and run summaries.

## Higher-order context ablation

The new optional context hierarchy learns which target colony co-occurs with a
pair of direction-tagged adjacent specialists. At prediction time the target
is masked; visible neighbor assignments query the pair table, with first-order
routes as a backoff. It does not consult image episodes or category labels.
Two independent strict-mask 600/200 Testing runs showed lower MSE than
first-order context: 0.013524 vs 0.015763, and 0.013674 vs 0.015989. It won on
84.5% and 90.5% of images; paired 95% bootstrap intervals for the change were
[-0.002743, -0.001764] and [-0.002836, -0.001783]. Pair motifs provided
91.0% and 92.25% of predictions. This is reproducible low-level patch-context
improvement on image-level splits. It is not evidence of brain anatomy
semantics, diagnosis, patient-level independence, or fungal equivalence.

The optional receptor-feedback loop was ablated on the same 400/200
Training-only development split with identical seed and image indices. It had
0.15189 RMSE enabled and 0.15196 disabled; the difference was negligible. On
the separate Testing confirmation, feedback-enabled RMSE was 0.15369 versus
0.15360 for the earlier feedback-disabled fit. It also prevented every
correction-gated suppression during training (0/9,600 suppressed). This does
not justify always-on receptor modulation. Paired image-level intervals
included zero on both splits: development MSE delta (on minus off) -0.000021
with 95% interval [-0.000092, 0.000056], and Testing delta +0.000028 with
interval [-0.000060, 0.000111]. The option remains available but defaults off;
surprise-updated context associations remain enabled.

Those historical receptor-feedback ablation scores used the pre-correction
masked-patch evaluator and are not leakage-safe; the feedback option remains
off by default until re-tested under the corrected protocol.

The novelty-threshold ablation used the same 400 unlabeled training images and
the same 200-image Training-only validation set, with identical seed and a
256-colony limit. Raising the threshold from the default `.012` to `.03`
reduced novelty overflow from 5,026 to 2,511 patch presentations while the
held-out context MSE changed from 0.023092 to 0.022983. Paired per-image
bootstrap difference (`.03` minus `.012`) was -0.000109, 95% interval
[-0.000364, 0.000124], so the apparent gain is inconclusive. At `.06`, overflow
fell to 175, but context MSE worsened to 0.023591; paired difference was
+0.000499, 95% interval [0.000069, 0.000908], and only 29% of images improved.
This shows that simply accepting more patches relieves capacity pressure but
eventually damages predictive specificity. Keep `.012` as the default until a
consolidation mechanism is evaluated; `.03` is a useful capacity trade-off
candidate, not a proven accuracy improvement.

The novelty-threshold and prototype-fusion comparisons above also used the
pre-correction evaluator. They remain engineering observations about overflow
and runtime, but held-out prediction deltas need to be repeated with target
masking before they can select a learning configuration.

Prototype fusion remains opt-in and disabled. Earlier thresholds were explored
on the pre-correction evaluator; their prediction results are superseded and
must be rerun under target masking before making any claim about their effect.

## Leakage-safe multi-scale experiment

The benchmark now masks each held-out target in preprocessed grayscale before extracting
edge/texture cues and matching visible patches. On two matched 400-image
unlabeled fits with 200 held-out Training images each, the single-scale model
had context MSE 0.024275 and 0.023166. Multi-scale cues lowered this to
0.022502 and 0.021594: paired per-image MSE changes were -0.001773 (95% CI
[-0.002255, -0.001347]) and -0.001572 (95% CI [-0.001972, -0.001197]). It
improved 74.5% and 77.5% of held-out images, respectively. Novelty overflow
fell from 5,026 to 2,854 on seed 20261007 and from 4,975 to 2,742 on seed
20261008. Source-pixel and encoded-cue overlaps were zero; no diagnosis labels
were used. Those runs used foreground crop/contrast normalization before
masking, so only local feature leakage was blocked. The later fixed-
preprocessing runs below remove that image-wide-statistic caveat.

The full-data training CLI now defaults to fixed full-frame preprocessing,
multi-scale cues, and pair-conditioned context motifs. The comparative
benchmark keeps hierarchy opt-in for matched ablations. The core `VisualConfig`
flags remain explicit for API and checkpoint compatibility. The feature combines native edge/texture responses
with 13x13 neighborhood statistics while keeping intensity unchanged. This
does not build a learned hierarchy; it is a fixed broader receptive field that
must now be tested on clean Testing images and richer MRI data.

A prior confirmation trained on 600 unique unlabeled Training images and
evaluated 200 Testing images after excluding previously scored paths. It used
the foreground crop/normalization mode and is historical comparative evidence,
not the strict fixed-preprocessing score. The newer strict runs are reported
above and in `HIERARCHICAL-CONTEXT-ABLATION.md`.

The held-out split was source-pixel and encoded-cue disjoint, but the dataset
does not provide verified patient IDs. It is therefore not patient-independent.
The earlier local-context runs on Testing images used the superseded evaluator
and are retained only as history. The project suite passed 226
tests after hierarchy, rollback, and checkpoint migration coverage were added. The earlier NumPy masked
retrieval axis bug has been fixed and covered by those tests.

## Full-corpus fit with the updated hierarchy

One unsupervised pass processed all 7,200 image files (115,200 patches) using
fixed preprocessing, multi-scale cues, and pair motifs. The model stored 6,878
unique canonical visual episodes, learned 102,657 pair-motif connections from
374,400 update events, and passed its built-in save/load, resume, resource,
label-field, and erasure checks. On a 256-image seen-set sample, encoded-cue
identity recall was 256/256. Mean reconstructed grayscale RMSE was 0.08386,
but external nearest-source identity matching was only 74/256 (28.9%). The
finite 256-colony bank recorded 15,061 novelty-overflow patch presentations.
Thus the run proves complete ingestion and exact cue recall, while exposing
weak reconstruction-based source retention and a meaningful capacity ceiling.
It does not measure unseen-scan generalization: the fit includes both data
partitions. The summary, 85 MB checkpoint, and visual source/reconstruction/error
panel are in the full-corpus output folder.

## Changes needed before claiming image learning

- Keep exact episode lookup, but report it strictly as seen-cue recall and never
  use it as an intelligence or medical accuracy score.
- The pair-conditioned three-patch context hierarchy is implemented and
  improved the strict-mask metric in two replications. The separate specialist
  bank still saturates: the 7,200-image run had 15,061 novelty-overflow patch
  presentations and 102,657 pair-motif connections. Next address capacity,
  decay, uncertainty, and hierarchy at higher image resolution; do not assume
  that the current motif count corresponds to meaningful biological structures.
- Continue testing the new local predictive relations on other datasets and
  patient-level splits. The local surprise-updated links exist. Prediction-error
  modulation of receptor updates is implemented as an opt-in ablation, but it
  showed no consistent transfer improvement, so it is disabled by default.
  Next, calibrate that plasticity on a separate development set and test whether
  it improves adaptation without damaging retained patterns.
- Continue beyond the fixed 32x32 cue plus three-patch motif layer with
  resolution pyramids, broader cross-patch prediction, and eventually 3D data.
  The current thumbnail is too destructive for a credible brain-scan claim. Real MRI work
  needs patient-linked volumes and sequence metadata, patient-level splits,
  external-hospital evaluation, and expert labels for any diagnosis readout.
- Keep biological claims testable: compare dynamic growth/transport to fungal
  time-series measurements separately from the image-learning benchmark.

No single bug fix can make this prototype understand any image in one pass.
One-pass exact storage is attainable and already present; transfer, semantic
recognition, and diagnosis require an objective and evaluation that the current
architecture does not contain.
