Xception
========

The backbone retains the original convolution, separable convolution,
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

The Xception entry in ``configs/config.yaml`` uses ImageNet weights at
``${WEIGHT_ROOT}/xception-b5690688.pth``. Run
``uv run python engine.py fit --model xception``. Training selects the best
checkpoint on the source validation split.

*Created by Codex.*
