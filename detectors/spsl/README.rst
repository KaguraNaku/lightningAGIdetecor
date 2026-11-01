Independent SPSL rewrite
=======================

SPSLTask composes the native four-channel Xception with ordinary whole-batch CE
training and unified evaluation. The local benchmark intentionally uses original
full Xception, despite the method name's shallow-learning reference. Production
code imports no legacy detector or optional model packages.

RGB preprocessing uses cubic resize to 256 pixels and mean/std 0.5. Inside the
model, the normalized RGB channels are averaged arithmetically, transformed by
fftn over (-1, -2), reduced to phase angle, reconstructed as exp(1j * phase),
inverse-transformed over the same dimensions and reduced to its real component.
That phase-only signal becomes the fourth input channel. No luminance weights,
extra normalization, amplitude information or detach is introduced. Input
gradients through the phase branch remain available.

ImageNet initialization preserves the native replacement order and RNG draws.
The old RGB conv1 is removed from the body import, a new four-channel conv1 is
constructed and each new channel receives the RGB weight mean. There is no extra
3/4 or variance scaling. Two-dimensional pointwise kernels and old missing BN
counters are translated explicitly; unexpected body keys or shapes fail.

Supply local init_weights, a strict trained-detector initial_checkpoint, or an
explicit initialization=random. The common converter and self-contained task
restore are supported. Saved construction retains four input channels, while
saved preprocessing remains RGB. Missing files never cause random fallback or
automatic downloads. The FFPP example keeps native batch 32 and optimizer
defaults, but the original config trains on FF-FS. Remake manifests and source
validation do not reproduce that sample population or selection protocol.

Native-resolution batch-2 checks compare actual local definitions with the
rewrite for preprocessing, phase, classifier-visible features, logits, CE,
gradients, BN buffers and one Adam update. A separate actual ImageNet check
compares the adapted stem, all model state and outputs with a seeded untrained
head. Unit tests cover spectral behavior, input gradients, stem adaptation,
strict conversion and native-resolution train/save/restore/evaluate. Historical
trained-detector scores and full-dataset reproduction remain unverified. Use
tools/bounded_run.py for model checks and tools/test_cpu.py for isolated serial
regression. Artifact links and commands are in configs/migrations/validation.json.
