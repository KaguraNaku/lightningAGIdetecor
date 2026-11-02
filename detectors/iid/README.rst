IID implicit and explicit identity detection
============================================

IIDTask composes a method-owned identity dataset/collator and Trainer with an
independent512-channel Xception and native IResNet50. Production code imports no
DeepfakeBench checkout. Existing baseline Xception modes remain unchanged.

Native mathematics and modes
-----------------------------

The implicit Xception adjusts its final2048 features to512 channels, pools them
and retains last_emb BEFORE subtracting the explicit512 embedding. Its binary
classifier is Linear512->2 on that difference. Explicit input is bilinearly
resized256->112 with align_cornersFalse. IResNet50 has stages[3,4,14,3], native
constructor/reinitialization ordering and a25088->512 FC. The complete detector
has399 parameter tensors;398 have gradients, including237 in the explicit
network. Only the explicit final features.weight is frozen. Constructor eval()
does not freeze the explicit branch: parent train() resets its mode normally.

The final explicit features BN executes only when batchsize>1, including eval.
Preserve this B1/B2 behavior rather than making inference batch-independent.
The native fp16=False explicit trunk disables enclosing CUDA autocast, while
its FC/final BN remain outside that context. CPU/GPU AMP equivalence has not
been established by the FP32 CPU checks.

Binary loss is CE of8*sigmoid(logits) after subtracting0.2 from each true-label
slot. It is neither ordinary CE(logits) nor BCEWithLogits. Identity loss uses
1000-class ArcFace, normalized embeddings/kernel, cosine clamp[-1,1], scale64
and margin0.4, with no CIFP/distributed extension. Signed EIC is mean cosine for
label1 MINUS mean cosine for label0, despite misleading source variable names.
The train objective is binary+.05*identity+.1*EIC. Native test loss omits identity.
Prediction remains softmax of the original TWO logits, class1 (higher is fake).
L2 normalization retains literal division without epsilon; empty-class means
and zero embeddings can produce NaNs. The Trainer records the complete failing
batch and source/RNG receipts and stops rather than repairing the objective.

IID's optimizer retains all399 ordered parameters, including its frozen one;
absent gradients prevent that parameter from receiving Adam/decay updates.
Best/last snapshots preserve full model, ArcFace kernel, optimizer, seeds and
RNG. Exact optimizer/RNG training resume is not exposed. Native defaults are
11 inclusive sweeps, Adam2e-4/decay5e-4, no scheduler and whole batches. Remake
source validation, checkpoint selection and optional early stopping are an
explicit selection policy rather than original full-experiment reproduction.

Data and checkpoints
-----------------------

Training accepts integer metadata.identity_index in[0,1000), or the native
integer after the last underscore in the image's parent folder. Example
001_002/0.png has identity2 for either binary label. Labels are never inferred
from the identity. Explicit metadata supports relocated/Parquet inputs; native
IID's tuple-path parser does not support Parquet itself. All train rows remain
in manifest order before the engine's configured sampler; no class balancing,
identity renumbering, real-only filter or dropping of failed requests occurs.

The native per-frame augmentation pipeline is retained during training only;
native train.py uses the ordinary base dataset for testing, without identity
supervision/augmentation. Images use native OpenCV RGB/cubic256 decoding and
mean/std0.5. A decode failure retries source0 with its actual identity AND label.
Requested slots and actual sources/occurrences remain separately recorded. A
second decode failure stops explicitly instead of native unbounded recursion.
A replacement can cross classes and leave a one-class EIC batch; that remains
a diagnosed nonfinite source objective, not an automatic balancing operation.

Supply original Xception init_weights AND explicit_weights for fresh training,
or explicitly select initialization: random for a constructed smoke fixture.
Full initial_checkpoint import and standalone state restoration are strict;
both branches, the unused native CE object and ArcFace kernel remain. Inference
restoration opens neither original initializer. Construction/preprocessing/loss
recipes are serialized and checked, including conditional BN and trainability.

Validation scope
------------------

experiments.detector_rewrite.iid_check compares every constructor tensor, mode,
trainability and Python/NumPy/Torch RNG; B1/B2 evaluation; outputs/losses/image
and all398 parameter gradients; full BN/model/Adam updates; negative objective
boundaries and strict score restoration. Use --task-adapter to construct through
IIDTask and its real optimizer. experiments.detector_rewrite.iid_data compares
20 native dataset/augmentation seeds and cross-label first-source replacement
against the actual base decoder and isolated original dataset method bodies.

The available reference uses actual Xception ImageNet and explicitly CONSTRUCTED
IResNet50 weights (seed2026), not recognition-pretrained or historically trained
IID weights. This distinction remains in receipts. Bounded CPU checks preserve
native RGB256/identity112 geometry at a declared miniature batch2/3; they do not
prove native batch32 or full-data training. All jobs use at most3GiB RAM, zero
swap, two CPU cores and hidden GPU; data audits use1536MiB. Full scientific
reproduction, recognition-pretrained alignment, historical trained scores and
GPU alignment remain unverified. Lightning dual experts and unified online
partial_fit/remove_batch remain unconnected.
