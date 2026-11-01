Independent LoRA rewrite
========================

The adapter implements the existing ExperimentTask protocol. LoRADetector owns
CLIP vision and a two-logit head; ImageClassificationTrainer handles plain
cross-entropy and whole-batch Adam/SGD; each task owns its preprocessing/state.
Production code does not import DeepfakeBench or loralib. Reference source hashes
are packaged in configs/migrations/source_manifest.json.

The reference is the local lora_detector.py implementation: CLIP-L/14, LoRA on
k/v/q/out attention projections, rank 16, alpha 16, frozen base/bias parameters,
and a trainable binary head. Preprocessing is the local RGB cubic resize and
CLIP normalization, with the local classification augmentation composition.
It is deliberately different from the archived Effort augmentation profile.

Mode switches and checkpoint state
----------------------------------

The rewrite preserves loralib's split training algebra and merged evaluation
algebra. Evaluation computes the merged matrix without changing the base
Parameter. The saved schema always says weight_mode=unmerged. Repeated eval,
train and save/restore cycles cannot double-apply the adapter.

A legacy import must explicitly set legacy_weight_mode=merged or unmerged.
For a merged state the importer subtracts the recorded delta once; inversion
can carry floating-point rounding from the original in-place merge. Exact
training resume from a merged legacy state is not claimed. Every supplied key
is then loaded strictly; no missing pretrained backbone is silently ignored.

Conversion uses the task's importer and the shared checkpoint envelope::

    python3 tools/bounded_run.py --output /absolute/new/conversion -- \
        .venv/bin/python -m detectors._shared.checkpoint \
        --config configs/experiments/lora_ffpp.yaml \
        --input /absolute/path/to/legacy.pth \
        --output /absolute/path/to/new-model.pth --legacy-weight-mode merged

The resulting model.pth is evaluated by experiment.py evaluate. The explicit
merged/unmerged choice must match the legacy writer's actual save mode.

Construction from a new checkpoint uses saved CLIP geometry and all weights,
without downloading the old backbone or accessing the reference checkout.
Cross-entropy objectives, trainable parameter names and Adam updates are covered
by independent differential audits in experiments/detector_rewrite/verify.py.
Constructed-state audits are not historical checkpoint or dataset reproduction.

Resource-bounded use
---------------------

For CPU commands, use tools/bounded_run.py with an absolute, new output path.
For example, from the repository root::

    python3 tools/bounded_run.py --output /absolute/new/check -- \
        .venv/bin/python -m unittest discover -s tests -p test_lora.py -v

The runner enforces 3 GiB memory, no swap, two CPU cores, host/disk reserves and
a finite timeout. CUDA is disabled by default. Optional GPU entry points have
an explicit PyTorch allocator cap (CUDA context overhead is additional).

configs/experiments/lora_ffpp.yaml preserves the native rank, batch, loss and
optimizer settings but uses the separately named remake-source-validation-v1
selection protocol. No full dataset training/reproduction is implied by the
unit or bounded differential checks. Check the migration ledger for later
detectors; the full 42-method expansion is not complete.
