LSDA independent model stage
===============================

This stage provides the default local LSDA mathematical model and two lazy
backbones. It does NOT provide the grouped dataset, native sampler transport,
Trainer or Task factory yet. LSDA therefore remains pending in the42-registration
ledger. Constructed-model and CPU component checks do not imply a complete
experiment pipeline or trained-detector performance reproduction.

Native default construction
------------------------------

Four fake-domain EfficientNet-B4 encoders and a separate B4 student each retain
the original static380-padding stem and unused1000-way ImageNet classifier.
Each wrapper adds a fresh1792->512 Conv1x1. This differs from the older remake
B4 baseline, which discards the pretrained stem/classifier. The local B4 trunk
implementation is reused without those wrapper replacements; production imports
no efficientnet_pytorch or DeepfakeBench. Explicit initialization imports the
ENTIRE B4 state, including stem and1000-way head, with only missing BN counters
initialized as in the source. Missing/wrong assets never silently become random.

The source real_encoder:null path constructs a RANDOM IResNet100, not a trained
identity teacher. Stage repeats3/13/30/3, kernel initialization, spatial pooling,
unused512*7*7 classifier and final1D BN remain. Its layer1/2/3 weights and final
features.weight are frozen. The stem/layer4 remain trainable; frozen stages
retain train-mode BN updates and pass derivatives to the stem. The content
encoder executes five times, once per domain, in original order. No whole-branch
freeze, eval override or distillation-target detach is introduced. The unused
identity classifier/final1D BN never execute in this spatial variant.

Whole grouped model and objectives
------------------------------------

Training uses Gx5x3x256x256 in canonical real/DF/F2F/FS/NT order. For each domain,
run content first, then its fake encoder when domain>0. Fake features stack as
Gx4x512x8x8; content stacks asGx5x512x8x8. The student observes all5G images in
ONE whole batch. Domain-specific models and the student are trainable.

Literal source augmentation concatenates groups along the channel dimension,
so domain means and hardest examples are8x8. It chooses one rule per domain,
then iterates512 spatial channel maps, drawing scalar(s) per channel. Preserve
interpolation, extrapolation, Gaussian noise and the first-two-CHANNEL difference
transform. Do not replace these with sample-wise/domain-wise corrections inferred
from the paper comments. Hard-example distance scalars detach via native CPU
torch.tensor construction. Python/NumPy/Torch RNG order is retained.

Mixup draws per-group torch.randperm4, alpha=torch.rand1*1.5+.5 and a Beta sample,
then assigns into zeros_like. Augmented/mixed features concatenate to1024 channels
and pass through the native trainable512-channel fusion stack. Domain CE uses
the original real content plus unaugmented fake features. Distillation is real
student-to-content MSE plus fake student-to-fused-feature MSE, with LIVE targets.
The signed real constraint is cosloss(real,NT)+cosloss(real,F2F)-cosloss(real,FS)
-cosloss(real,DF), using norm+1e-8, sum and final scalar mean in native order.
Overall loss is domain +.5*binary CE +distillation +real constraint. Invalid
training output transport now raises instead of the source broad except silently
returning zero; valid objectives are unchanged. This guard is a declared boundary.

All2577 parameter tensors survive. Of2153 trainable tensors,2140 receive gradients.
Native optimizer groups include every parameter, including424 frozen tensors and
13 unused trainable tensors. Only2140 have Adam state after the audited update.
Inference executes only student features/binary head. Native optional observers
and its feat=prob output remain available in forward_outputs; forward(images)
returns two logits for the eventual common Task interface.

Bounded activation storage
-----------------------------

engine.saved_tensors.DiskSavedTensors is an opt-in CPU saved-activation context.
It does not alter batch shape, dtype, forward operations, gradients or modes.
Persistent model storage stays resident. Large saved activations spill to disk;
views share a file by C++ weak storage identity, version and dtype, and unpack
restores exact shape/stride/offset and validates every value hash. Live alias
version checks retain in-place failures. Cache hints release clean file pages
without modifying file contents. It is inactive for existing detector training.

Call cleanup only after successful backward without retain_graph, after saving
transport metadata; it removes OWNED technical scratch. Preserve scratch on
failures. Inputs, model/optimizer/gradient states and experiment metrics are
scientific artifacts and remain preserved. The successful native reference uses
the configured whole batch10 (G2),3GiB hard RAM, zero swap, CPU2 and hidden GPU.
Its first600-second audit timed out AFTER saving gradients; scratch IO and
memory pressure were observed, and the failed lineage and its scratch survive. A
fresh1200-second capped audit completes all native forward/backward/Adam gates
and removes only successful scratch. Limits and model mathematics were not relaxed.

Evidence and work remaining
------------------------------

Original primitives record20 seeds, full latent values/derivatives and RNG;
source sampler/collate behavior and the CPU class-definition CUDA failure have
separate raw receipts. Transport checks compare ordinary native latent derivatives
and repeated native residual/BN/dropout derivatives/state/RNG, including strided
inputs and expected invalid in-place failures. Full native/independent model
gates compare constructor, modes/freezing/RNG, every output/loss/image and2140
parameter gradients, BN/model/Adam state and strict restored scores. See migration
ledgers for actual passing receipts, failures, source hashes and commands.

An additional ORIGINAL-data reference executes the selected sampler/item/collate
methods on an explicit320-row inventory (2groups x5domains x32slots), recording
20 seeds at whole batch10 with full image tensors, indices, labels and RNG.
The original slot31->30 same-video replacement succeeds; persistent missing-video
decoding recurses116 times before the audit's temporary recursion limit produces
a preserved RecursionError. Only the class-definition hardware query is bridged
for CPU. Native constructor/split discovery and full worker/RNG protocol remain
unverified. These are original-source reference receipts, not an independent
dataset implementation or an integrated training workflow.

The grouped sampler/data/Task/Trainer remain UNCONNECTED. The native sampler must
shuffle groups, draw one shared slot for their five videos, and preserve batch-
index collection before augmentation. Its dedicated brightness/flip/downscale/
sharpen/JPEG profile must not be replaced by the common image augmentation.
Neighboring-slot replacement, requested/actual frame provenance, invalid-group
filtering and full-source validation isolation require explicit integration and
runtime data checks. The original dataset's eager CUDA query fails on CPU and
its persistent decode recursion needs a recorded bounded repair. The alternative
Xception constructor is also unverified/broken in the local source and is not
silently substituted. No official LSDA experiment config is exposed at this stage.

Historical trained scores, GPU alignment, full51-sweep experiments, exact training
resume and complete LSDA OO workflow remain unverified. Lightning dual experts
and unified partial_fit/remove_batch entry points remain unconnected.
