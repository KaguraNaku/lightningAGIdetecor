GenD CLIP
=========

``GenDTask`` composes ``GenDDetector``, a CLIP encoder and ``GenDTrainer``.
The shared preset trains the encoder's LayerNorms and the LinearNorm head, uses
the author's PIL augmentation, and optimizes CE plus 0.5 uniformity and 0.1 label
alignment. It trains on FF++ with logical batches of 128, AdamW at 0.0003,
zero weight decay, a 10-epoch cosine restart period and a 30-epoch limit.

Run ``uv run python engine.py fit --model gend``. For output gradient caching in FP32,
set ``models.gend.fit.gradient_cache_microbatch_size`` in
``configs/config.yaml``. The objective and optimizer step still operate on
the full logical batch. Add ``--timing`` for a run-level ``timing.json``.

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
The complete model forward disables autocast on CPU and CUDA. Tests check
actual gradients and updates for trainable tensors.

The Task reports the fake-class probability from its two logits. The author
training profile applies PIL augmentation during training and uses CLIP
normalization for training and evaluation.

Checkpoint identity and imports
--------------------------------

The shared training preset starts from the cached OpenAI CLIP ViT-L/14
backbone with a fresh head. The Task can also load the published
``yermandy/GenD_CLIP_L_14`` model at revision
``891ce014a0308386c4d7d25b3dcf436a22db5504``. Custom geometry requires explicit
random initialization or a compatible raw checkpoint.

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
Large temporary workflow snapshots use the system temporary directory. Bounded
test logs identify their paths. Successful temporary test fixtures are removed.

The reference CE/Adam audit used train batch 24, evaluation batch 32 and
disabled augmentation. The shared training preset selects checkpoints on the
source validation manifest. To resume, pass a fitted ``model.pth`` to
``--checkpoint`` and keep the paired ``best.pth``, which stores the selected
epoch's optimizer, scheduler, gradient-cache settings and RNG state.

*Created by Codex.*
