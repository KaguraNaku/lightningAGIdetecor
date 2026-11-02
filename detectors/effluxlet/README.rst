Independent Effluxlet rewrite
==============================

EffluxletTask composes a lazy CLIP-L/14 backbone, EffluxletDetector and its
method-owned trainer. Production imports neither DeepfakeBench nor
pytorch_wavelets/PyWavelets. The common pipeline supplies manifests, source
validation, selection, checkpoint receipts and image/video evaluation.

Native model math
------------------

The native recipe is RGB224, CLIP-L/14 hidden1024/depth24, sidecar rank32 and
four Haar levels. All96 q/k/v/out projections retain frozen main weights/biases
as buffers plus learned A/B residuals. A uses Kaiming uniform and B uses normal
std1e-6; the residual scale is1. Inputs cast to float32 for sidecar arithmetic
and results cast back, exactly as the source does. Other CLIP parameters freeze;
the wavelet branch and classifier remain trainable. ModuleDict aliases are
retained when present. The wavelet branch is constructed before sidecar draws,
then the head, preserving the checked constructor ordering.

The branch retains four-level Haar/zero filtering, directional CDC with
dilations1..4,9 output channels, three directions, native channel adaptation,
grouped RMS scaling, global pooling and MLP mappings. All four mapped vectors
sum into pooled CLS. Source enhancement indices only determine depth; there is
no per-layer token injection. The source's unused inverse constructor consumes
no RNG and is not retained as model state. Its dormant DTCWT mode is not part
of this selected fixed recipe.

Haar coefficients were extracted from the original PyWavelets1.8 environment.
Production implements separable analysis and the original custom synthesis
backward order directly in Torch, including odd-size zero extension/cropping.
It retains all four named filter buffers. The reference runs actual selected
pytorch_wavelets numerical definitions with a narrowed metadata adapter for
captured coefficients and verified zero-mode output lengths. Filtering and
gradient arithmetic remain unchanged. Component audits compare
even224/odd223 outputs and input gradients, plus every branch parameter gradient.

The loss is CE plus100 times mean absolute cosine similarity between each
frozen main matrix and B@A. The full matrix product and native reductions remain;
it is not replaced by a lower-memory surrogate. Scoring softmaxes binary logits.
Augmentation defaults enabled with source CLIP mean/std and the shared native
image-transform implementation. Deterministic workflow fixtures disable it;
these are not a reproduction of original sample/augmentation history.

Explicit training-entry repair
-------------------------------

The actual generic train.py chooser reads optimizer.adam.lr. The Effluxlet YAML
instead contains all.lr5e-6 and wavelets_branch.lr1e-4, so the unmodified chooser
raises KeyError('lr'). That failure is executed and preserved. The migration
explicitly interprets these as disjoint groups: all other trainable parameters
at5e-6 and wavelets_branch parameters at1e-4. Adam beta0.9/0.999, eps1e-8 and
weight_decay5e-4 retain the source configuration. The bridge exposes the two
rates and weight decay; the remaining Adam constants stay at the native recipe.
Frozen parameters are omitted. This is a documented interface/configuration
repair, not proof that the unusable original training entry was reproduced.

Reference training uses the same explicit group interpretation and verifies
group membership, all optimizer tensors and actual updates. Shared selection
still requires source-domain validation with disjoint identities. Source
epochs0..15 imply the default16 sweeps. The FFPP example retains batch24 and
eight frames/video, while using remake manifests and validation selection.
The source monument_update is an unused pass stub; it does not establish an
online update capability.

Initialization and restoration
--------------------------------

Default initialization pins cached OpenAI CLIP-L/14 revision
32bd64288804d66eefd0ccbe215aa642df71cc41, offline-only. init_weights can select
the same complete CLIP safetensors locally. The loader selects the vision state
and discards text state. Old cached position_ids are checked against the exact
derived arange table before omission, because current CLIP makes them
nonpersistent. Other key/shape mismatches fail strictly. Random initialization
requires initialization: random and is never a missing-weight fallback.

initial_checkpoint imports the complete native Effluxlet tensor layout. Engine
checkpoints retain the vision configuration, fixed recipe, preprocessing,
weights, sidecar main buffers and Haar filters without the original cache/source
files. Trainable tensors receive owned storage when restored by assignment.
There is no extra phase/bank state in this active method. Exact optimizer/RNG
resume is not exposed by the task.

Evidence and limits
--------------------

Sequential full-native batch1 audits agree exactly for initialized state and
trainability, all branch/fused/logit scores, CE/regularizer, every gradient,
complete frozen/updated model tensors, Adam state and restored scores. The
frozen pretrained checkpoint is hardlinked into the archive. Initial and updated
trainable overlays reconstruct each complete model without duplicating its
trunk. Inputs, raw outputs, gradients, optimizer state, source snapshots, full
tensor fingerprints, configs, seeds and environment receipts are preserved.
Only SHA256-verified identical immutable duplicates share hardlinks.

Six tests cover analytical Haar bands/energy gradient, aliases and residual
math, disjoint optimizer groups, strict conversion, invalid position metadata
and full L/14 training/evaluation with selection of an earlier best epoch after
an injected negative validation score. These selection fixtures are not accuracy
estimates. Large temporary workflow snapshots use /var/tmp and failed fixtures
are retained. All model checks use bounded_run.py: one CPU job, RAM3072MiB,
swap0, CPU2 and host RAM/disk reserve4096MiB. Historical trained-detector scores,
GPU numerical alignment and full-dataset reproduction remain unverified. Raw
paths and reproducible commands are in configs/migrations/validation.json.

A later complete-suite attempt hit the unchanged3GiB cap while copying the
selected frozen CLIP/SVD state after training. Its full best/last checkpoints,
history and configs are preserved under persistent /var/lib/codex-experiments-
naku, with the original temporary path retained as a symlink. Selected weights
now use strict assigned mapped storage; trainable parameters alone are cloned.
This method has no mutable BatchNorm buffers. A storage-ownership regression
and full native L/14 workflow pass under the same cap. Actual pretrained model,
gradients, optimizer update and restored scores still agree byte-for-byte with
the preserved native reference. Geometry, rank, batch and mathematics are
unchanged. The failed run remains distinct from the corrected evidence.
