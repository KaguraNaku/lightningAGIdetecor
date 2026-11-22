Effort in the remake pipeline
============================

``EffortTask`` implements the shared experiment contract and composes an
``EffortDetector`` and an ``EffortTrainer``. The detector returns two logits;
the task exposes fake probability. Dataset manifests, decoding, coverage and
final image/video metrics use the existing pipeline.

The algorithm baseline is YZY-stack/Effort-AIGI-Detection author commit
``96f5dea2b534d400cfd7003f053c7e93c8e16461``, as archived and used by the local
effort-full-suite-20261101 experiment. Source hashes are saved in task state.
The active objective in that version is cross entropy. The modified
DeepfakeBench working-tree variant with orthogonal/spectral losses is not the
baseline of this adapter.

Training
--------

From the repository root::

    uv sync --locked
    uv run python engine.py prepare-data --dataset FaceForensics++ --split train --frames-per-video 8
    uv run python engine.py prepare-data --dataset FaceForensics++ --split val --frames-per-video 32
    uv run python engine.py prepare-data --dataset Celeb-DF-v2 --split test --frames-per-video 32
    uv run python engine.py run --model effort

Training uses the author's Albumentations 1.3.1 augmentation composition, Adam
(lr 0.0002, betas 0.9/0.999, eps 1e-8, weight decay 0.0005), and only trains
the trailing SVD components plus the classifier head. For ViT-L/14 this is
198754 trainable parameters. Each attention matrix retains 1023 frozen leading
components and trains the remaining singular component. The effective batch is
32, processed in microbatches of 8 with sample-weighted gradient accumulation.
The default training seed is 1024, precision FP32, with at most 10 epochs.
OpenCV is pinned to 4.11.0.86: version 5 changes a few cubic-resize output pixels.
The Effort runtime config explicitly disables TF32 for convolution and matmul.
Use the same setting in data-only evaluation configs when replaying FP32 scores.

Training and validation collators are independent objects. Validation has no
training augmentation. Model selection uses source-domain validation video AUC
after averaging frame probabilities, with patience 2 and minimum improvement
0.0001. The best weights are restored before the shared engine saves model.pth.
Test inputs never enter the trainer. The probability threshold is 0.5.

The config uses remake's uniformly spaced manifest sampler. It does not claim
the author's first-eight-JSON-frame protocol or the historical run's frozen
sample membership. For historical comparisons, import the exact recorded sample
membership into manifests and preserve the selection seed separately from the
training seed. PIL decoding is retained; the migration audit compares pixels
to the author's OpenCV decoding on its selected source images.

Legacy weights and offline evaluation
-------------------------------------

Convert a checkpoint once, then use the shared evaluation entry point::

    uv run python -m detectors.effort.checkpoint \
        --input /absolute/path/to/author/best.pth \
        --output /absolute/path/to/new/remake-effort.pth \
        --local-files-only
    uv run python engine.py evaluate \
        --model effort \
        --checkpoint /absolute/path/to/new/remake-effort.pth

Conversion supports a raw author state_dict or a training envelope containing
state_dict, with optional DataParallel module prefixes. It requires the CLIP
config (cached when using --local-files-only), but no backbone weights or SVD
initialization. State loading is strict, including the author's frozen SVD
components. The converted checkpoint contains geometry, preprocessing, score
policy, provenance and all weights. Subsequent restoration requires neither
the original checkpoint nor a Hugging Face download.

Set model.initial_checkpoint to an absolute author checkpoint path to start a
new training run from those weights. This is a warm start, not exact resume.

Artifacts and verification
--------------------------

The fit directory preserves protocol.json, environment.json, history.json,
best.pth and last.pth. Each completed epoch has training receipts and skipped
IDs, plus validation predictions, coverage and metrics. Failed invocations and
partial receipts remain available through the shared engine's run directory.
Snapshots include optimizer/scaler/RNG states for inspection. Exact training
resume, including prefetched worker RNG state, is not implemented.

Small model behavior and engine integration are checked by tests/test_effort.py.
The bounded real-checkpoint audit is experiments/effort_migration_20261101/verify.py;
invoke it as a module with --help for its checkpoint, source, prediction and
output arguments. It retains pixels, logits, predictions, environment and
failure reports. It checks migration equivalence, not generalization quality.
