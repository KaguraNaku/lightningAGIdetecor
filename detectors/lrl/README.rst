LRL local relation learning on the unified pipeline
===================================================

LRLTask composes a synchronized RGB/IDCT/mask dataset, a method-owned Trainer
and an independent dual-B4 model. Production code imports no DeepfakeBench.
Model mathematical parity and deliberate repairs to the broken original data
entry are different gates. No full original data-entry equivalence is claimed.

Native model and objective
----------------------------

Two original B4 wrappers retain static380-padding schedules, fresh unpadded
RGB stems and unused1792->2 classifiers. The IDCT branch constructs a normal
three-channel wrapper before replacing its stem with one channel. Preserve
that otherwise discarded RNG draw. Both branches use actual B4 ImageNet trunk
weights when init_weights is supplied; their fresh stems/heads remain fresh.

Three RFAM modules use native in-place sigmoid attention on56,160,1792 channels.
Low/mid maps resize by nearest .25/.5 and concatenate with high features into
2008x8x8. Unfold2/stride2 produces16 patch tokens, each8032-dimensional. Literal
no-epsilon cosine maps to(sim+1)/2. Removing diagonal entries in row order yields
240 classifier values. Classification is Linear240->128/ReLU/Dropout.2/Linear2.

The reverse decoder stages are2008->256->256->128->64, with nearest2x, Conv3,
InstanceNorm(no affine/running statistics) and in-place ReLU. A final bilinear2x
with align_cornersTrue produces256 masks; Conv64->1 and sigmoid remain. The
decoder/similarity path also executes during native inference; it is not omitted.
All875 parameter tensors are retained;871 receive gradients. Four unused
encoder classifier tensors remain trainable but without gradients or Adam state.

Ground-truth relations use native4x4 chunk order, float32 means and detached
CPU scalar construction followed by1-L2 distance, including the diagonal.
Loss is similarity MSE plus sigmoid-mask BCE plus CE, equally weighted and in
native addition order. No epsilon, loss reweighting or BCEWithLogits replacement
is introduced. Zero feature norms retain nonfinite native relations.

Explicit data transport corrections
-------------------------------------

Original LRLDataset unpacks four fields from a base method that currently returns
five. Its squeeze(2).permute(2,0,1) mask expression fails for actual HW/HWC1
arrays, and missing-mask float64 zeros cannot be passed to float32 BCE. Four
train/test, present/missing-mask cases reproduce these errors in the unmodified
original bodies. Their outputs, configs and tracebacks remain preserved.

The remake adapter names source fields, transports grayscale HW/HWC1 directly
as1HW, and casts missing zero masks to float32. Missingness, native dtype and
the applied transport are recorded. Supplied-mask values remain unchanged;
native grayscale/linear256 resize, division by255 and synchronized geometric
augmentation remain. Missing masks are not silently discarded, made positive
or replaced by boundaries. Explicit metadata.mask={path,sha256:optional} binds
relocated sidecars; otherwise frames->masks paths are derived for images.
Parquet inputs may supply explicit sidecars or native no-path zero supervision.
File contents and missing-file absence are pinned before workers start.

RGB pixels use native cubic256 decoding and mean/std .5. Training augmentation
follows the original base body, passing a mask only when it contains positives.
IDCT is computed from the SAME augmented raw uint8 RGB: gray conversion BEFORE
float32 cast, cv2.dct, the active native high-pass threshold .33, cv2.idct, then
(value/255-.5)/.5. The source actively passes r1=.33,r2=.33; do not substitute
the unused .66 default. Frequency values may be outside[-1,1] and stay unclipped.

Decode errors stop with diagnostics instead of the broken legacy recursion;
there is no silent source replacement, relabeling or skipped training row.
Evaluation uses the same RGB/IDCT pairing without training augmentation. The
engine's existing predict_batch route carries explicit auxiliary inputs. Direct
predict(Tensor) recovers exact uint8 pixels only from the task's native FP32
normalization grid; arbitrary continuous/half-rounded RGB requires explicit
aligned IDCT. It never silently quantizes unsupported inputs.

Training and verification scope
--------------------------------

The ordinary image Trainer has a batch-loss hook for aligned auxiliary inputs;
its default delegates to the prior compute_loss(images,labels). LRL overrides
the hook, retains whole batches and reuses fit/source validation/checkpoint
selection. Native defaults are Adam2e-4/decay5e-4,41 inclusive sweeps and no
scheduler. Remake validation/early stopping is a declared selection policy.
Strict standalone state includes both branches, decoder, unused heads and the
complete model/preprocessing/repair recipe. Original initializer files are not
needed for inference restoration; exact optimizer/RNG resume is not exposed.

experiments.detector_rewrite.lrl_check compares every constructor/RNG/mode,
output, loss, RGB/IDCT and871 parameter gradients, full BN/model/Adam update
and strict score restoration against the unmodified native mathematical model.
Use --task-adapter to construct/restore through the real Task and optimizer.
experiments.detector_rewrite.lrl_data checks40 declared repaired-bridge cases
(20 train,20 test), comparing original base/IDCT/normalization/target primitives
with all corrected RGB, IDCT, masks, labels, relation targets and RNG. That
reference explicitly applies the same three transport repairs. It does not
pretend the original complete dataset entry ran successfully.

Bounded CPU checks use full256 geometry at a declared miniature batch2 with
actual ImageNet trunks and fresh untrained heads, at most3GiB RAM, zero swap,
two CPU cores and hidden GPU. They do not reproduce native batch32 or full-data
training. Historical trained scores, GPU alignment and full41-sweep experiments
remain unverified. Lightning dual experts and unified partial_fit/remove_batch
remain unconnected.
