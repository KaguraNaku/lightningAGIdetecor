Independent Meso4 and MesoInception4 rewrite
===========================================

MesoTask composes a native backbone, binary logit detector and shared whole-batch
CE trainer. Select variant=meso4 or variant=meso4Inception. The backbone registry
loads only the requested implementation; there are no legacy-checkout imports,
automatic weight downloads or additional optional model libraries.

Both variants use RGB cubic resizing to 256 pixels, mean/std 0.5 and native
random initialization. Their local configs disable augmentation. Meso4 reuses
bn1 in its first two blocks and bn2 in its last two; MesoInception4 reuses bn1
after both ordinary convolutions. Inception branches retain their native
concatenation order, dilation, lack of intermediate activations and historical
Incption checkpoint spelling. The classifier retains Dropout2d on 2D features.
Current PyTorch warns that this usage is deprecated; this rewrite deliberately
preserves the tested operation and records its environment.

Raw state dictionaries (including module.-prefixed envelopes) load strictly.
The common checkpoint converter writes a self-contained engine envelope:

  .venv/bin/python -m detectors._shared.checkpoint \
    --config configs/experiments/meso4_ffpp.yaml \
    --input /absolute/source.pth --output /absolute/new-model.pth

Run model commands through tools/bounded_run.py. Both config examples retain
native batch 32, 10 epochs and Adam defaults, but use remake manifests and source
validation selection. They select FaceForensics++ for illustration whereas the
local legacy configs select FF-NT for training. They do not reproduce the
original sample population, batch order or model-selection protocol.

Reference/candidate fixtures use seeded constructed weights, native resolution
and batch 2 with a recorded common dropout RNG state. Fresh initialization,
preprocessing, features, logits, probabilities, CE, gradients, all model state
including shared BN buffers, and one Adam update agree exactly. CPU tests also
exercise train/save/restore/evaluate for both variants. Historical trained
checkpoint scores and full-dataset reproduction remain unverified. Artifact
links and commands are in configs/migrations/validation.json.
