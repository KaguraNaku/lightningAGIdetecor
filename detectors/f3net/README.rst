Independent local F3Net rewrite
===============================

F3NetTask composes four-band frequency-aware decomposition with a twelve-channel
full Xception and shared whole-batch CE training. The checked local detector has
FAD only; LFS and other paper components are not present in its source and have
not been added. Production code imports no legacy detector or optional model
library. Native module/checkpoint names retain FAD_head and its fixed parameters.

The DCT is orthonormal, with its own fixed transpose tensor. Each band applies a
fixed inclusive mask plus 2*sigmoid(learnable)-1, followed by inverse DCT. Band
boundaries are 0, size//2.82, size//2 and size*2; adjacent endpoints overlap, and
learnable residuals can affect frequencies outside their base band. The RGB
outputs are concatenated low/middle/high/all into 12 channels. Fixed bases and
DCT parameters stay frozen; all four residual matrices are trainable. Native
constructor RNG draws include both normal samples per residual parameter.

Construction first creates the RGB backbone, including dropout 0.5 and its
initialization draws. ImageNet import accounts strictly for the body, then
replaces conv1 with four RGB copies each divided by 4.0. The frequency modules
are initialized after this replacement, preserving seed alignment. Explicit
random construction retains the same structural replacement order. Missing files
fail without random fallback or automatic download. Strict raw conversion and
standalone restore preserve geometry, masks, DCT tensors, stem and dropout.

Native preprocessing is RGB cubic resize to 256, mean/std 0.5, no augmentation.
The FF++ configuration uses batch 32, the original optimizer defaults and source
validation. The original configuration uses FF-NT. Training uses whole batches.
The source manifests define sample membership and order.

Native-resolution batch-2 fixtures check fresh initialization, preprocessing,
FAD output, classifier-visible features, logits, CE, gradients, BN buffers and
one Adam update with replayed dropout RNG. Separate actual ImageNet checks cover
all initialized model state and predictions, including the twelve-channel stem
and seeded untrained head. Unit tests cover spectral boundaries, residuals,
initialization, strict conversion and native-resolution train/save/restore/
evaluate. Historical trained scores and full experiments remain unverified.
Run model checks with bounded_run.py and complete CPU regression with test_cpu.py.
