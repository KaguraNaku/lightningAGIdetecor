Effort
======

``EffortTask`` implements the shared experiment contract and composes an
``EffortDetector`` and an ``EffortTrainer``. The detector returns two logits;
the task exposes fake probability. Dataset manifests, decoding, coverage and
final image/video metrics use the existing pipeline.

The implementation follows YZY-stack/Effort-AIGI-Detection commit
``96f5dea2b534d400cfd7003f053c7e93c8e16461`` and uses cross entropy.
The Task saves the source file hashes.

Training
--------

From the repository root::

    uv sync --locked
    uv run python engine.py run --model effort --test-inputs ffpp_test celeb_df_v2_test

Training uses the author's Albumentations 1.3.1 augmentation composition, Adam
(lr 0.0002, betas 0.9/0.999, eps 1e-8, weight decay 0.0005), and only trains
the trailing SVD components plus the classifier head. For ViT-L/14 this is
198754 trainable parameters. Each attention matrix retains 1023 frozen leading
components and trains the remaining singular component. The effective batch is
32, processed in microbatches of 8 with sample-weighted gradient accumulation.
The shared preset uses seed 1024, FP32 and at most 11 epochs.
OpenCV is pinned to 4.11.0.86: version 5 changes a few cubic-resize output pixels.
Effort defaults to TF32 off. Use ``--no-tf32`` to set this explicitly.

Training and validation collators are independent objects. Validation has no
training augmentation. Model selection uses source-domain validation video AUC
after averaging frame probabilities, with patience 3 and minimum improvement
0.0001. The best weights are restored before the shared engine saves model.pth.
Test inputs never enter the trainer. The probability threshold is 0.5.

The config samples evenly spaced frames when preparing a manifest. The author
uses the first eight frames from each JSON entry. For historical comparisons,
import the exact recorded sample membership into manifests and preserve the
selection seed separately from the training seed. PIL decoding is retained.
The pixel audit compares it with the author's OpenCV decoding.

Author weights and offline evaluation
-------------------------------------

Convert a checkpoint once, then use the shared evaluation entry point::

    uv run python engine.py import-checkpoint --model effort \
        --input weights/effort-author.pth \
        --output outputs/effort-converted.pth
    uv run python engine.py evaluate \
        --model effort \
        --checkpoint outputs/effort-converted.pth

Conversion supports a raw author state_dict or a training envelope containing
state_dict, with optional DataParallel module prefixes. It requires the CLIP
config, which the shared preset loads from the local cache. The checkpoint
supplies the SVD components. Strict state loading includes the author's frozen
SVD components. The converted checkpoint contains geometry, preprocessing, score
policy, provenance and all weights. Subsequent restoration requires neither
the original checkpoint nor a Hugging Face download.

Set ``models.effort.settings.initial_checkpoint`` to start a new training
run from author weights.

Artifacts and verification
--------------------------

The fit directory preserves protocol.json, environment.json, history.json,
best.pth and last.pth. Each completed epoch has training receipts and skipped
IDs, plus validation predictions, coverage and metrics. Failed invocations and
partial receipts remain available through the shared engine's run directory.
To resume a fitted model, pass ``model.pth`` to ``fit --checkpoint`` and keep
the paired ``best.pth``. It contains the selected epoch's optimizer, scaler
and RNG state. Use nonpersistent loader workers for resume.

*Created by Codex.*
