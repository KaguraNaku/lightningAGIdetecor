Independent local RECCE rewrite
===============================

RECCETask composes a distinct Xception encoder, decoder, per-position graph
reasoning and reconstruction-guided attention. Shared whole-batch CE fitting,
source validation, strict raw conversion, standalone state and frame/video
evaluation retain the existing pipeline. Production imports neither the old
detector nor timm.

The active encoder follows local timm legacy_xception.tf_in1k: four activation
nodes, fan-out Kaiming convolution resets, and a retained unused 1000-class fc.
The new lazy xception_timm backbone shares verified convolution/block primitives
but preserves this different layout and constructor order. The wrapper also
constructs an unused DeepfakeBench Xception first. That branch, its adjustment
path and binary head remain in strict state and receive no gradients. Source
wrapper pretrained settings do not initialize that unused branch.

init_weights explicitly initializes the active encoder from xception-43020ad28.pth,
not the different DeepfakeBench xception-b5690688.pth. Imports validate the full
encoder state including its unused ImageNet head. Missing weights fail without
random fallback or network downloads. Training from scratch requires explicit
initialization: random. Constructor ordering retains all RNG draws before the
new binary head, attention, reasoning and decoders are created.

Training adds a CPU-drawn per-sample Bernoulli gate times Gaussian noise with
std 1e-6 and clips to [-1,1]. Evaluation applies no noise. Decoder features at
ratios 2 and 4 feed graph reasoning after encoder block7. Per-position stack,
linear weighting, softmax, batch matrix product, complementary gates and sigmoid
reprojection preserve source ordering; the spatial loop is not vectorized.
Absolute reconstruction residual guides attention after block8 with bilinear
align_corners=True resizing. Dropout 0.2 is retained in decoder input, graph
reprojection, guided attention and the final pooled feature.

The checked local wrapper consumes only CE logits. Reconstruction and three
correlation matrices remain observable auxiliary outputs; no reconstruction or
contrastive loss is added. Decoder and graph parameters still receive gradients
through the CE classification path. Correlation uses unrestricted squeeze(), so
the batch-one auxiliary correlation is scalar; that native boundary is retained.

Preprocessing is cubic RGB resize to 256, mean/std0.5 and disabled augmentation.
The example retains native batch32 and optimizer settings but uses FFPP manifests
and remake source validation instead of the original FF-NT sample/order/selection
recipe. Bounded checks use batch2; no full-protocol reproduction is claimed.

Native-resolution constructed-state audits compare fresh state, preprocessing,
reconstruction, correlations, pooled features, logits, CE, gradients, buffers and
one Adam update with training RNG replay. Actual ImageNet initialization is
checked separately with seeded untrained binary heads. Historical trained-detector
scores and full-dataset reproduction remain unverified. Receipts are linked in
configs/migrations/validation.json. Use tools/bounded_run.py for model execution
and tools/test_cpu.py for complete CPU regression.
