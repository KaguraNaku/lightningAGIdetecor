Independent Xception rewrite
============================

The rewritten backbone retains the original convolution, separable convolution,
residual block and BatchNorm layout, including checkpoint names and the unused
adjustment branch. The ordinary original/shallow_xception classification paths
are supported. Other specialized channel-adjustment variants require a separate
verified specification rather than silently changing reference dimensions.

XceptionTask composes the backbone/logit detector, local 256-pixel RGB cubic
preprocessing with mean/std 0.5, and a whole-batch classification trainer. The
trainer rejects microbatch_size because splitting a batch changes BatchNorm
updates. It never silently substitutes random initialization for a missing
pretrained file. Supply init_weights, an initial_checkpoint, or explicitly
request initialization=random.

The ImageNet initializer supports legacy non-ZIP serialization, expands 2D
pointwise kernels, excludes the 1000-class head and accounts explicitly for
missing old BatchNorm counters and classifier/unused adjustment parameters.
Unexpected body keys or shapes fail. Trained detector imports and self-contained
restoration load strictly.

configs/experiments/xception_ffpp.yaml requires a real local initialization path.
It selects models using source validation, under remake-source-validation-v1;
this is not an assertion of original sample/protocol reproduction. Use
tools/bounded_run.py for execution. Do not change batch size or claim a full
training run simply to avoid a resource limit; retain the failed bounded run.
