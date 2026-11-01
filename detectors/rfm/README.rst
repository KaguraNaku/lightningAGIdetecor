Independent RFM rewrite: checked local recipe
============================================

RFMTask composes a native Xception/logit detector and a method-owned RFMTrainer.
The trainer owns FAM construction, random rectangles and CE computation. The
inference model performs no FAM, backward or random augmentation. Production code
imports no old sources or optional model packages.

The local backbone returns (logits, spatial features). Its FAM targets the second
result: classifier-activated feature channel 1 minus channel 0, summed through an
all-ones backward seed over all samples/spatial locations. The backbone runs in
eval mode during FAM. Input gradients become abs/max-over-RGB saliency. FAM also
leaves backbone parameter gradients, while its classifier head has none. The
standard source trainer clears gradients before forward and subsequently adds CE
gradients. This rewrite preserves that accumulation. Under source AMP, FAM is
unscaled and only CE is scaled; this ordering is retained, though the recorded
numeric checks cover FP32 CPU. SAM is not exposed by the shared trainer.

For each 256-pixel image, NumPy's descending argsort ranks the FAM, Python random
selects up to three unmasked points and rectangle offsets bounded by 120, and a
single torch.rand_like draw replaces masked RGB values with uniform [-1, 1]
noise. The original rectangle boundaries, empty-slice behavior and full-tensor
noise draws are retained. No extra post-FAM gradient clear, input-only autograd,
class-logit target, mask threshold or clipping is introduced. The backbone is put
back in train mode for the augmented classification forward. Training geometry
is explicitly restricted to native RGB 256; inference remains ordinary Xception.

Native Tensor.cuda hardcodes the FAM backward seed's device. Production allocates
the same all-ones seed on target.device. The CPU reference changes only that
allocation with a temporary recorded CUDA-to-CPU redirection. The original CPU
failure remains in the preflight record. NumPy, Python and Torch RNG states are
recorded/replayed independently by the audit. Pre-CE FAM gradients, post-CE total
gradients, saliency, augmented pixels and the update are retained separately.

Initialization is explicit local ImageNet, strict raw trained weights, or
initialization=random. The saved rfm_recipe identifies the checked feature-target
and retained-gradient behavior. Standalone restore and common conversion are
strict. Ordinary loader augmentation settings do not disable RFM's training
augmentation. The FFPP example keeps native batch/optimizer settings but uses
remake manifests and source-validation selection; original data membership and
selection are not asserted. Historical detector scores, GPU numeric alignment
and full-dataset reproduction remain unverified. See the migration validation
ledger and execute checks through bounded_run.py / test_cpu.py.
