Independent GenD CLIP adapter
==============================

GenDTask composes GenDDetector, a lazy CLIP encoder and the shared whole-batch
trainer. The selected DeepfakeBench recipe is yermandy/GenD_CLIP_L_14; its cached
configuration specifies LinearNorm. The port supports linear and LinearNorm
heads explicitly. External GenD's DINO and Perception alternatives are not
connected by this CLIP recipe. Production never imports the GenD checkout or
modifies sys.path.

Native semantics
-----------------

The native CLIP-L/14 has RGB224, hidden dimension1024, 24 layers and 16 heads.
Its pooled CLS features feed the classification head. LinearNorm applies L2
normalization before the linear layer; linear does not. The retained
768-dimensional visual_projection is checkpoint state but is unused by this
forward. It remains frozen.

The source unfreezes feature-extractor parameters whose names contain norm or
ln, and freezes all other feature-extractor parameters. The full head remains
trainable. At native geometry this yields102 trainable parameter tensors.
The complete model forward disables autocast, retaining the source's CUDA
FP32 policy; the adapter also disables CPU autocast. Tests verify actual gradient
presence rather than relying solely on requires_grad flags. CE and Adam remain
the source definitions. Native Adam enrolls frozen parameters too; the shared
trainer filters them. Named optimizer state and every updated/frozen tensor
are checked to establish equivalent active updates. Exact resume across these
different optimizer index layouts is not claimed.

The ordinary task prediction path softmaxes two classification logits. Shared
RGB cubic resize and the source CLIP mean/std are used; the external image
processor is not used by DeepfakeBench's tensor forward. Augmentation defaults
to disabled, as in gend.yaml.

Checkpoint identity and imports
--------------------------------

Default loading is offline-only and pins the published model revision
891ce014a0308386c4d7d25b3dcf436a22db5504. The default profile retains the actual
cached head configuration. No download or random fallback occurs when it is
missing. Explicit initialization: random is available for constructed fixtures.
Custom geometry requires that option or an explicit raw checkpoint.

initial_checkpoint accepts raw GenD keys (feature_extractor and model.linear)
or DeepfakeBench wrapper keys (model.feature_extractor and model.model.linear).
DataParallel module prefixes are handled before strict loading. Missing/extra
keys and incompatible shapes fail. Raw imports require an explicit head, because
the two head semantics have identical tensor layouts and cannot be inferred.
Both PyTorch weights-only files and safetensors are supported. Safetensors
dispatch uses the original filename before resolving HF's extensionless cache
blob symlink. Trainable tensors receive owned storage after assign-based loading.

The source wrapper unconditionally prepends model. when loading, even for its
own wrapped state_dict, so it fails to reload its own saved layout. The port
explicitly repairs that interface while preserving numerical model rules. The
unmodified reload failure and its traceback are archived. Standalone engine
checkpoints retain head, vision configuration, freeze/precision policy,
preprocessing and all model tensors, including the unused projection. They
restore without the external checkout, initial checkpoint or HF cache.

Evidence and resource limits
-----------------------------

Reference and candidate execute sequentially at full native geometry and batch1
using the published cached detector checkpoint, SHA256
d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833.
The raw checkpoint is hardlinked into the artifact archive so HF cache cleanup
does not lose it or require another1.2GB copy. Initial model bytes, freeze masks,
synthetic RGB preprocessing, published scores, CE, every gradient, named Adam
state, updated weights and restored scores agree exactly. Frozen tensors are
checked unchanged, and the source checkpoint hash is checked after training.

The oracle executes selected actual GenD/DeepfakeBench definitions with the
same CLIP library. Nested pretrained transport is replaced by meta construction
and complete explicit tensor assignment, avoiding discarded text weights and
network access. This establishes the loaded model trajectory; it does not
establish random-constructor RNG equivalence or historical benchmark results.
Raw inputs, scores, gradients and optimizer state are preserved. Updated
trainable tensors plus the immutable source checkpoint reconstruct the full
updated model without duplicating the frozen trunk; full tensor fingerprints
verify the complete result. Source snapshots, failures, seeds and environment
information are archived separately.

Tests cover both heads, normalization and zero features, freeze gradients,
autocast behavior, both raw layouts, HF symlinks, strict standalone restore and
the full L/14 training/evaluation workflow. Best selection is checked after an
injected negative second validation score; this tests control flow, not accuracy.
Large temporary workflow snapshots use /var/tmp and retain failed runs; bounded
test logs identify their paths. Successful temporary test fixtures are removed.

The example preserves train batch24/eval batch32, Adam lr2e-4/decay5e-4,
disabled augmentation and no scheduler. Eleven sweeps match source epochs0..10.
Remake manifests and validation selection differ from the full original
experiment. Historical benchmark checkpoint scores, GPU numerical alignment and
full-dataset reproduction remain unverified. All model checks use bounded_run.py:
one CPU job, RAM3072MiB, swap0, CPU2 and host RAM/disk reserve4096MiB. See
configs/migrations/validation.json for commands, run IDs and artifact paths.
