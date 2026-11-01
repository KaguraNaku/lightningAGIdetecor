Independent EfficientNet-B4 rewrite
==================================

EfficientNetB4Task composes an independent B4 backbone/logit detector with shared
whole-batch CE training, source-validation selection and unified evaluation.
Production code uses PyTorch directly and imports no efficientnet_pytorch or
torchvision modules. The registry loads it on demand. Native state names retain
efficientnet._blocks and the binary last_layer.

The local wrapper deliberately replaces the stem AFTER loading ImageNet weights
with a fresh unpadded Conv2d. Downstream padding remains static, calculated for
the original B4 canvas of 380 pixels, even though preprocessing resizes to 256.
This rewrite preserves that behavior. It also preserves 32 rounded MBConv blocks,
SE ordering, batch normalization epsilon/momentum, exact custom Swish backward,
drop-connect probability scaling and random masks, initialization RNG draws,
native part/end-point feature interfaces and optional outer classifier dropout.
The inner ImageNet dropout is not used by the wrapper's classifier.

Initialization must be explicit: provide init_weights for a local ImageNet file,
initial_checkpoint for trained raw detector weights, or initialization=random.
There is no missing-file fallback or automatic download. ImageNet import accounts
for every trunk key while excluding the deliberately replaced stem and 1000-class
head. Raw detector imports, common conversion and standalone restoration are
strict. Saved state includes geometry, preprocessing and the model
construction settings; original sources and weight paths are unnecessary for
restoration.

Original/original mode is supported. The local adjust_channel path maps features
to 512 channels but leaves its classifier at 1792 inputs; it is explicitly rejected
pending a separate specification. No hidden classifier repair is introduced.
The example keeps batch 32 and optimizer defaults but uses remake FFPP manifests
and source validation; the local legacy YAML selects FF-NT. It does not claim
original sample membership or training-protocol reproduction.

Native-resolution batch-2 fixtures compare seeded initial state, preprocessing,
features, logits, probabilities, CE, gradients, BN buffers and one Adam update
using the same recorded drop-connect RNG state. A separate actual ImageNet
initialization audit checks the fresh replacement stem explicitly. CPU workflow
tests cover strict import and native-resolution train/save/restore/evaluate.
Historical trained-detector scores and full experiments remain unverified.
Artifact paths and commands are in configs/migrations/validation.json. Execute
model checks through tools/bounded_run.py and retain any failed capped runs.
