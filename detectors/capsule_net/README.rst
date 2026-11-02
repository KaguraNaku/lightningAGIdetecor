Independent CapsuleNet rewrite
===============================

CapsuleNetTask composes CapsuleNetDetector, the lazy vgg19_capsule backbone and
CapsuleTrainer. Production code imports neither torchvision nor DeepfakeBench.
The shared pipeline supplies manifests, RGB decoding, validation selection,
checkpoint receipts and image/video metrics.

Native construction and features
---------------------------------

The native input is RGB256 normalized with mean/std0.5. VGG19 is constructed
with the same order of default Conv/Linear resets as torchvision's
pretrained=True path, which forces init_weights=False. Its discarded classifier
still consumes random draws before capsule construction. The independent
implementation retains that construction order, then discards unused layers.
The prefix includes indices0..18, including the third max-pool, and produces
256x32x32 features. The earlier static review's 64x64 description is corrected
in the delivery receipt; the original review remains archived.

VGG parameters remain trainable. The source only sets the initial prefix mode
to eval; parent train() changes it recursively. Its unused freeze_gradient
method does not establish frozen parameters. Both VGG gradients and its actual
Adam update are checked, rather than inferring training from requires_grad.

Ten separate branches use two Conv2d/BN/ReLU stages, spatial mean and sample
standard deviation, two Conv1d/BN stages and an eight-value view. Their stacked
output is squashed over ten input capsules. Convolution weights and BN scale
follow the source normal resets; convolution biases retain their default draws.
Routing weights have shape2x10x4x8. Two iterations normalize over input capsules
and squash over output coordinates. Native batch-one squeeze handling remains.
The wrapper disables routing noise and dropout in both train and eval; dormant
CUDA-only random branches are not added to the supported fixed recipe.

Loss and score semantics
-------------------------

Softmax over the two routed classes produces shape(N,4,2). CapsuleLoss sums
four CE terms on those probabilities, retaining the source's probability-input
CE arithmetic. The detached mean over four coordinates is the source cls
output. A second softmax produces its reported fake probability. Shared task
prediction and validation scoring apply that same second softmax; the method
trainer obtains its differentiable loss from classes, never the detached cls.
The cls output is not raw classification logits.

The native squash has no epsilon and gives NaN for zero vectors. Independent
source/candidate boundary observations preserve that behavior. No numerical
repair, VGG freezing, branch reduction or smaller spatial geometry is hidden in
this migration. Such changes would need a separate algorithm decision and audit.

Initialization and checkpoints
-------------------------------

Supply init_weights with the complete local ImageNet VGG19 file. Its entire
layout is checked strictly before the unused layers are discarded. Missing or
incompatible files fail explicitly; production does not download or substitute
weights. For constructed fixtures, explicitly select initialization: random.
That option retains pretrained-constructor RNG/layout but supplies constructed
weights, and is not an ImageNet reproduction claim.

Raw initial_checkpoint imports retain native vgg_ext, fea_ext and routing_stats
names and reject missing/extra tensors. Engine checkpoints include the fixed
construction recipe, preprocessing and weights. They restore under meta without
the original ImageNet/raw checkpoint or source checkout. There is no additional
bank/counter state in this active native method. Exact optimizer/RNG resume is
not exposed by the unified task.

Validation and limits
----------------------

Separate constructed-state and actual ImageNet-weight audits run at native
RGB256 and bounded batch2. Construction, preprocessing, feature shape, classes,
detached cls, scores, loss, every gradient, BN buffer and Adam state/update match
exactly. Batch1 and batch2 eval and standalone checkpoint scoring also match.
Initial weights, input RGB/tensors, intermediate outputs, full gradients,
optimizer state, updated model states and source hashes/snapshots are archived.
Only byte-identical immutable duplicates are hardlinked, retaining all paths.

Tests cover sample std, zero-vector behavior, analytic probability-input CE,
strict conversion without the original file, score semantics, full training,
image/video metrics and selection of the earlier best epoch after a negative
validation result. Validation-score injection tests selection behavior and does
not provide an accuracy estimate. Full raw audit paths and reproducible commands
are recorded in configs/migrations/validation.json.

The example keeps native batch32, disabled augmentation, Adam lr8e-5 and
weight_decay5e-4, no scheduler and eleven sweeps (source epochs0..10). It uses
remake FFPP manifests and source-validation selection, differing from the
original multi-dataset sampling and selection. Historical trained-detector
scores, GPU numerical alignment and full-dataset reproduction remain unverified.
Model runs must use tools/bounded_run.py; current verification uses CPU only,
one model job, RAM3072MiB, swap0 and CPU2. The discarded full VGG classifier is
transiently allocated to retain native random draws; only the prefix is stored.
