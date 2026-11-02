FWA with native generated/original pairs
========================================

FWATask composes the independent original Xception FWADetector, FWAGenerator,
FWAPairCollator and FWATrainer. Generation belongs to training collation;
validation/evaluation retain ordinary RGB images and common image/video metrics.
Production imports no DeepfakeBench code and creates no import-time predictor.
Install the optional ``fwa`` dependencies; the audited environment uses dlib20,
albumentations1.3.1 and scikit-image0.26. Other detectors do not import dlib.

Native generation rules
------------------------

Decoded RGB is resized to256 with OpenCV cubic interpolation. A lazy per-worker
dlib20 predictor reads its explicitly supplied, SHA256-identified81-landmark
asset. The source uses its first68 points. Canonical51-point coordinates,
source similarity/SVD order (including its V.T placement), default affine
interpolation and multi-face alignment stay intact. Only the first aligned
face is used for blending. Scale is chosen by Python random from0.2..0.8;
default resizing, Gaussian5x5, five hull choices, std20 piecewise deformation,
restriction and random mask blur retain the source's order. Warp-back uses a
separate positive-value mask per color channel, not an alpha feather repair.

The dataset's custom augmentations run even though the source global
use_data_augmentation flag isFalse. Original-view augmentation, generated-view
augmentation and discarded post-processing augmentation retain their exact RNG
consumption. Training post-processing swaps channels, then tensorizes and
normalizes by mean/std0.5. Evaluation does not apply this swap. No-face fallback
duplicates the augmented original and keeps its source label; both views still
participate in the model batch. Generation errors other than no-face are not
silently relabeled or skipped.

Each source produces generated+original views. Native collation concatenates
all generated views then all originals and Python-shuffles the whole list.
Source batchB therefore means model batch2B; CE/BN/Adam see that complete batch.
No splitting, substituted ordinary classification batches or independent
per-view augmentation is used. Defaults retain Adam2e-4/decay5e-4, no scheduler
and eleven sweeps. All160 Xception parameter tensors remain trainable;156 get
gradients, while unused adjust_channel tensors remain serialized without them.

Explicit transport changes
---------------------------

The native __getitem__ guesses labels from absolute-path component6 and fails
for arbitrary paths. The remake uses manifest real0/fake1 labels. Audits first
use compatible native paths and confirm label identity, then retain the original
arbitrary-path failure. This is a disclosed data transport repair. Inventory,
decoding, source-validation selection and final evaluation use remake contracts;
they do not reproduce the original full FF-NT sampling protocol.

Coverage and samples.jsonl count source images once. views.jsonl records both
shuffled model views; generation.jsonl records parent identities, generation
parameters, pixel hashes, fallback decisions and complete batch Python/NumPy RNG
states. Coverage separately includes model_images, generated_views,
original_views and no_face_fallbacks. Loss averaging uses actual model images.
Source and view identities/labels are validated before the optimizer update;
failed batches retain partial receipts. Native source batch16/effective32 is in
the example; bounded audits use source2/effective4 without splitting it.

State and ownership
---------------------

Raw detector imports are strict. Task state retains the full native pair recipe,
preprocessing, model/head/BN state, source/init identity and training-predictor
path/hash/size. Inference restoration never opens that training-only asset.
Further fitting requires the same hashed asset, which can be relocated via an
explicit predictor_path. Worker pickling clears C++ detector/predictor owners;
each worker loads its own predictor lazily. No C++ object or active graph is
serialized. Exact optimizer/RNG training resume is not exposed.

Validation scope
------------------

Native data audits use actual dlib inference and successful/no-face fixtures.
They compare alignment, all five hulls,20 stochastic mask seeds, complete paired
pixels/labels/order and post-generation RNG exactly. Constructed and actual
ImageNet-weight model audits compare initialization, effective4 inputs, scores,
CE, all parameter/image gradients, BN/model state, named Adam state/update and
restored scores byte-for-byte. ImageNet audits have seeded untrained binary
heads and do not establish historical trained FWA scores.

Seven unit tests cover discarded augmentation RNG, pair/fallback provenance,
whole-batch BN/Adam against a manual loop, corrupt labels before update, lazy
worker ownership, strict standalone import and native training/best selection/
evaluation. The workflow uses cached actual alignments for its exact offline
image fixture; actual dlib inference is verified separately by the native audits.
Injected validation0.9/0.8 is control-flow evidence, not measured accuracy.

The complete CPU suite passes177 tests in27 serial capped groups. Model audits
and FWA workflow tests peak below2GiB; the controlled dlib build peaks2.28GiB.
The full suite reaches3GiB with memory reclaim, zero swap and no OOM or GPU use.

Raw states, inputs, gradients, optimizer, source snapshots, public fixture origin,
dependency build and negative results live in persistent
/var/lib/codex-experiments-naku with home aliases. configs/migrations/validation.json
links commands and receipts. Model jobs use one CPU process group, at most3GiB
RAM, swap0, CPU2 and hidden GPU, while preserving4GiB host RAM/disk reserves.
The dlib build uses two compiler jobs and disables CUDA. A host-reserve rejection
before a lightweight fixture extraction is retained; no model ran in that job.

Historical trained-detector benchmark scores, GPU numerical alignment, exact
training resume and full dataset reproduction remain unverified. Lightning
dual-expert fusion and unified partial_fit/remove_batch remain unconnected.
