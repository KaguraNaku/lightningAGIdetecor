Independent SRM rewrite
=======================

SRMTask composes two separately constructed Xception backbones, fixed residual
filters, spatial guidance, two dual cross-modal attention modules and channel
fusion. SRMTrainer owns the fixed raw-logit cosine-margin objective. Shared
whole-batch fitting, source validation, checkpoint conversion and frame/video
evaluation retain the existing pipeline. Production imports no legacy code.

Large fit snapshots are written once. best.pth atomically links the immutable
last.pth inode; later atomic replacement of last leaves the selected best intact.
Snapshots should be treated as immutable artifacts. Completed batch gradients
are released before validation/checkpointing, and the optimizer is released
before loading the selected weights. These changes retain all snapshot fields
and training math while reducing simultaneous memory and duplicate writes.

Source behavior is preserved even where it looks unusual. Separate filters use
repeat(axis=0), grouping whole kernels rather than interleaving all three per
channel. Residuals are clipped to [-3,3]. Spatial attention filters an already
filtered image again. Cross attention uses a shared key projection, independent
value projections and linear maps fixed at 16x16 positions. Both gamma parameters
start at zero. Fusion multiplies by channel attention without a residual sum;
its native direct-child initialization loop does not reset the nested convblk.
Default constructor draws and subsequent Xavier/Kaiming resets retain their
original order, including unchanged convolution and linear biases.

Both branches import actual ImageNet weights separately before the later branch
or attention modules are constructed. Unused adjustment branches, the unused
residual classifier and Gaussian blur weight buffer remain in strict state.
Noise and blur objects are never invoked by native features; no extra Gaussian
augmentation is added. Checkpoints retain fixed geometry and loss parameters.

Training subtracts 0.45 from the target-class raw logit, scales both logits by
30 and applies cross entropy (cos margin, gamma=0, t=1). There is no feature or
classifier normalization. The rewritten selection mask uses bool instead of
the source's deprecated uint8; values and gradients are compared against the
actual source loss. Inference and validation use ordinary raw-logit softmax.

RGB cubic resize to 256 and mean/std 0.5 are retained, with augmentation disabled
by default. The FFPP example selects different manifests from the old FF-F2F
training config and uses remake source validation. It does not assert original
sample membership, order or model-selection reproduction. Native batch 32 is
documented in that example; bounded checks use batch 2 without microbatching.

Validation receipts live in configs/migrations/validation.json. Constructed-state
audits deliberately set both gamma parameters to 0.25 to exercise key, value and
linear gradients. Actual ImageNet initialization is checked separately with the
native zero gamma and seeded untrained heads. Neither check establishes trained
historical detector scores or full-dataset reproduction. Model checks must use
tools/bounded_run.py; full CPU regression uses tools/test_cpu.py.
