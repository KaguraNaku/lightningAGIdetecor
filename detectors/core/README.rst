Independent CORE rewrite
========================

CORETask composes the native Xception backbone, a logit-only detector interface
and a method-owned CORETrainer. The detector also exposes forward_outputs for
training: spatial features, their non-inplace positive average-pooled CORE
representation and classification logits. The classifier preserves the native
in-place feature activation; reference feature checks account for this aliasing.

The objective is native ConsistencyCos: normalize the pooled representation,
pair the first floor(N/2) rows with the following floor(N/2) rows, compute cosine
similarity, apply mean squared error to ones at fixed coefficient 1.0 and add CE
over every sample. Pairing is positional and is not changed according to labels.
For odd batches, the final sample participates only in CE. A singleton has no
defined native consistency mean; it raises a clear error without dropping,
duplicating or regrouping samples. Whole batches must be retained, including
when a memory budget is tight. The shared trainer's compute_loss hook lets CORE
own this objective while reusing receipts, optimizer handling and selection.

Native defaults are RGB cubic resizing to 256 pixels, mean/std 0.5 and disabled
training augmentation. Initialization is explicit: supply local Xception
ImageNet init_weights, a strict full-detector initial_checkpoint, or request
initialization=random. Saved tasks preserve model structure, preprocessing and
the fixed objective coefficient; they restore without old sources or weight
files. The common checkpoint converter is supported. No legacy imports or
optional model packages are used in production.

core_ffpp.yaml retains batch 32, Adam defaults and 10 epochs, but uses remake
FFPP manifests and source-validation selection; the local legacy YAML trains on
FF-NT. It does not reproduce old sample membership or selection. Do not replace
batch-halves pairing with paired-image augmentation based only on a method name.

The native-resolution batch-2 differential compares the actual local detector,
ConsistencyCos and backbone definitions with the rewrite, including auxiliary
features, full loss, gradients, BN buffers and one Adam update. Unit tests also
cover positional/odd pairing, singleton failure, strict checkpoint conversion
and the native-resolution unified workflow. ImageNet initialization is a separate
gate; the binary head is seeded rather than trained in that check. Historical
trained-detector scores and full-dataset reproduction remain unverified. Use
tools/bounded_run.py for model checks and tools/test_cpu.py for serial capped
regression. Retained results, failed audit attempts and commands are linked in
configs/migrations/validation.json.
