Independent SIA rewrite
=======================

SIATask composes an independent EfficientNet-B4 with three active information
attention sites and learned, pooled residual routes. Shared whole-batch CE
fitting, source validation, strict raw conversion, standalone checkpoints and
frame/video evaluation retain the existing pipeline. Production imports no
legacy detector or optional EfficientNet library.

Native SIA manually calls all 32 B4 blocks without drop-connect arguments, even
in training. It does not use the baseline helper that enables stochastic depth.
The original 380-pixel static-padding schedule and fresh unpadded RGB stem are
retained at 256-pixel input. Attention sites follow blocks 5, 9 and 21. Two max
pools fix the residual routes at 32x32 and 16x16, so the task requires 256 input.

Intra/inter distances are detached. The inter-distance concatenates five full
channel-offset tensors then views the five-element range inside channels before
summing; it is deliberately not replaced by a conventional offset stack. Both
normalization means span the whole batch, channels and spatial positions.
Spatial coefficients are sigmoid(distance/mean/2)+1, and channel coefficients
are spatial mean(-log(exp(-distance/mean/2)+0.1))+1. Gradients through attention
gains are absent, while gradients through scaled features and the three learned
residual projections are retained. The route projections consume the detached
normalized distances returned by attention, not attended feature tensors.

There is no epsilon in the normalization denominator. Degenerate zero-distance
features produce native NaNs; unit coverage records that behavior rather than
silently repairing it. The shared trainer rejects nonfinite losses explicitly.
Changing batch composition can change an image's prediction in evaluation as
well as training. Preserve loader grouping when comparing historical scores;
do not split a batch or claim batch-size-independent frame scores.

Only att1conv, att2conv and att4conv execute, but all seven attention modules and
the unused one-output linear head remain in strict state. Actual ImageNet import
retains native constructor RNG and the freshly replaced stem and binary head.
Missing initialization fails without random fallback or downloads. The saved
attention specification identifies detached gradients, batch-global mean,
distance layout and disabled drop connect.

Preprocessing is RGB cubic resize to 256, mean/std 0.5 and native augmentation
enabled by default. The example preserves batch 32 and optimizer defaults but
uses FFPP manifests and remake source validation rather than the original FF-NT
sample and selection recipe. It is not an original-protocol reproduction config.

Native-resolution batch-2 audits compare fresh state, preprocessing, all three
attention maps, features, logits, CE, gradients, BN buffers and one Adam update.
Actual ImageNet initialization is checked separately with seeded untrained heads.
Historical trained-detector scores and full-dataset reproduction remain unverified.
Validation receipts are linked in configs/migrations/validation.json; use
tools/bounded_run.py for model checks and tools/test_cpu.py for full regression.
