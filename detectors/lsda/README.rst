LSDA independent implementation
================================

LSDATask composes the native default detector, method-owned grouped Trainer,
explicit manifest dataset, sampler and collator. The common engine resolves
inputs and forwards an optional sampler hook after dataset adaptation; it has
no LSDA-specific branch. Training retains whole five-domain groups. Validation
and final evaluation use ordinary complete image manifests and the shared
image/video metrics. Migration status is gated by the recorded full regression;
workflow checks and mathematical alignment do not imply trained-score or full
experiment reproduction.

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

Grouped data and unified workflow
------------------------------------

LSDAInventoryBuilder derives group metadata from an existing complete train
manifest and explicit official FF++ split pairs. It follows native pair order,
emitting both directions, and preserves each video's32 existing rows in their
declared manifest order. It never selects new images or changes binary labels,
paths, storage or physical frame indices. Group-specific ids retain their exact
upstream source ids and manifest hash. Missing domains/slots, ambiguous video ids
and conflicting labels fail; unused source ids and upstream reconciliation
identities remain in the preparation receipt. Existing outputs are never replaced.

The remake FF++ adapter names categories youtube/Deepfakes/Face2Face/FaceSwap/
NeuralTextures; these are the preparation/preset map. A caller with FF-real/FF-DF/
FF-F2F/FF-FS/FF-NT metadata must supply an explicit category map. No alias or
fake-type renumbering is inferred. Prepare a fresh grouped manifest::

    .venv/bin/python -m detectors.lsda.manifest \
      --input /absolute/path/to/FaceForensics++-train-frames-32.jsonl \
      --pairs /absolute/path/to/FFpp/train.json \
      --output /absolute/path/to/FaceForensics++-train-lsda.jsonl

LSDADataset validates all five domains and32 slots per group, then LSDAGroupSampler
shuffles groups with Python RNG and draws one shared requested slot per group.
DataLoader collects each batch's indices before per-image augmentation. Sampling
uses the declared complete group inventory, replacing the source hardcoded720
group count. Do not treat320 inventory rows as320 observations per epoch: the
two-group reference yields10 sampled images. Per-epoch receipts distinguish
requested/actual sources, invalid excluded groups and unselected inventory slots.

Its dedicated brightness/flip/nearest-downscale+linear-upscale/sharpen/JPEG
augmentation is always active, as in the original getitem despite the source
use_data_augmentation:false setting. Original slot0/31 edges and interior random
neighbor recovery retain video/domain and RNG. Persistent failure now stops at
explicit max_decode_attempts with all attempted sources and RNG; original
unbounded recursion and its negative receipt remain preserved. Invalid label sets
are filtered by whole group; all-invalid input fails. A valid domain set in the
wrong order or with mixed requested slots fails rather than silently sorting.

Native and independent20-seed whole-batch data checks agree in every image value,
label, index and RNG, both manually and through create_input_loader. Sixty native
slot0/interior/31 recovery cases agree in pixels, labels, actual neighbors and RNG.
The prepared-manifest path separately replays20 native seeds. Spawn-worker tests
verify deterministic remake transport; native full worker/RNG protocol remains
unverified. Remake's private loader generator and worker seeding are explicit
policies, not a claim of the original training process's complete RNG trajectory.

The actual Task constructor, all2577-parameter optimizer,2140 gradients, BN/Adam
update and strict Task predict restoration also match the native batch10 model.
A separate constructed-random full workflow runs two native-geometry epochs,
real source validation, early stopping, best-weight restoration and final image/
video evaluation. Its min_delta2 selection probe forces deterministic second-
epoch non-improvement without replacing actual AUC calculation. Raw validation
metrics and predictions survive; this is a workflow test, not trained performance.

configs/experiments/lsda_ffpp.yaml exposes the default B4/IResNet100 route. Set its
actual B4 weight path, grouped manifest and durable output directory. Execute
training through tools/bounded_run.py with explicit timeout and resource budgets.
The whole-batch CPU workflow requires20GiB free for temporary activations and
retained checkpoints; successful activation scratch is removed only after backward.
REMAKE_EXPERIMENT_ROOT selects workflow-test artifact storage and must be passed
inside the bounded command with /usr/bin/env, because the service forwards an
explicit environment. The first workflow preflight failure is retained.

Explicit manifests replace native eager CUDA-name/hardcoded-drive and cross-split/
c40 discovery. Source frame discovery/selection is not claimed reproduced by
grouping an existing remake32-frame manifest. The alternative Xception constructor
is unverified/broken in the local source and is not exposed by the preset.

Historical trained scores, GPU alignment, full51-sweep experiments and exact
training resume remain unverified. Lightning dual experts
and unified partial_fit/remove_batch entry points remain unconnected.
