Independent ResNet34 rewrite
===========================

ResNet34Task composes a native backbone, binary logit detector and the shared
whole-batch CE trainer. The model retains RGB input, the [3, 4, 6, 3] residual
stages, stride placement, fan-out convolution initialization and BatchNorm
updates. It requires no torchvision import or compiled vision extensions.
The lazy backbone registry imports this implementation only when selected.

Sequential trunk keys (resnet.0 through resnet.7) and the binary fc head match
the local wrapper. The unused ImageNet head's initialization RNG draws are
retained so an explicit random initialization agrees with the source at the
same seed. Mode=adjust_channel saves the historical extra branch but leaves it
unused in forward, matching the local source. RGB cubic preprocessing uses
256 pixels, mean/std 0.5 and ordinary training augmentation.

The source unconditionally requests torchvision pretrained=True and ignores
the YAML pretrained path. This rewrite makes initialization explicit: supply
init_weights for a local ImageNet state, initial_checkpoint for a trained raw
detector state, or initialization=random. Missing weights never trigger a
download or random fallback. The ImageNet importer translates stem/stage keys,
ignores only the declared 1000-class head, initializes absent legacy BN counters
and strictly accounts for every trunk tensor. Raw detector imports and offline
task restoration load strictly. The common checkpoint converter is supported.

resnet34_ffpp.yaml retains native batch 32, Adam defaults and 10 epochs, but
uses remake FFPP manifests and source validation. The local old config trains
on FF-NT; the example is not an original-protocol reproduction. Run commands
through tools/bounded_run.py. A resource cap does not authorize changing native
BatchNorm batch semantics or starting an unattended full-dataset experiment.

The native 256-pixel constructed-state fixture uses batch 2. It compares the
actual local detector/backbone definitions and installed torchvision Python
definitions with the independent model; production code never imports those
reference sources. Separate validation uses the actual ImageNet V1 weights
specified by that torchvision source. See configs/migrations/validation.json
for individual gates, file hashes, resource receipts and raw artifacts.
Historical trained-detector scores and full experiment reproduction are
separate unverified gates.
