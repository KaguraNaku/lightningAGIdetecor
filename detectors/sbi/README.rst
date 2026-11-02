SBI real sources and three-logit classification
================================================

SBITask composes an SBIRealDataset view, SBIPairCollator/SBIGenerator,
SBITrainer and an independent original EfficientNet-B4 SBIDetector. The engine
only supplies an optional dataset-adaptation hook before sampling/workers.
Validation/evaluation retain their complete binary-labeled inputs. Production
imports no DeepfakeBench code, efficientnet_pytorch, torchvision or imgaug.

Literal source initialization and score
---------------------------------------

The checked YAML sets backbone_config.pretrained:true and num_classes:3.
The actual EfficientNet loader accepts that boolean through its cached model-
zoo branch and loads B4 ImageNet weights. Its stem and1000-class head are then
replaced by the native fresh unpadded stem and a fresh THREE-class head. The
independent B4 retains its native380 static-padding schedule at RGB256.

The detector additionally reads an Xception checkpoint using strictFalse.
Runtime accounting finds706 matched keys in the internal B4 import, zero
missing/unexpected; the outer Xception import matches zero tensor keys and
changes no model state. Production omits this useless read and explicitly
requires B4 init_weights or initialization:random. It never substitutes random
weights when a requested initialization fails. The literal unusable absolute
Xception path and its FileNotFoundError remain preserved.

All three logits participate in CE on labels0/1 and in softmax[:,1] scoring.
The third row receives nonzero gradients. Do not convert this recipe to a
two-class head or renormalize only the first two logits. Native mean/std are
ImageNet0.485/0.456/0.406 and0.229/0.224/0.225, distinct from the B4 baseline0.5.

Selected source-only training view
------------------------------------

SBIRealDataset selects label0 records BEFORE the sampler, batch sizing, decoding
and source coverage. Fake source rows remain in the immutable parent manifest
and are counted as excluded. source-selection.json binds the parent manifest,
selection counts and complete sidecar identities. Selected IDs/storage/video
identities remain the same; validation/test datasets are not filtered.

Real image records may use native frames-to-landmarks PNG/NPY sibling paths.
For other layouts or Parquet, supply explicit Sample.metadata.landmarks:

  {"path": "relative/to/dataset/root.npy", "sha256": "...", "coordinate_size": 256}

The sidecar must contain numeric finite68x2 or81x2 points in256-coordinate space.
No implicit rescaling/reordering is performed. Source int32 conversion is kept.
Hashes are fixed before sampling and checked again during loading; missing,
changed or malformed real supervision fails explicitly. Excluded fake rows do
not require image or landmark IO. Inference checkpoints require no sidecars.

Generation and ordering
------------------------

The native whole-face route retains four hull recipes, optional first68
truncation with probability0.25, source-or-target photometric augmentation,
affine plus elastic mask transformation and stochastic blend ratios. The
elastic IMAGE output is computed/discarded while its MASK is retained, as in
the source. Shared additional_targets augmentation synchronizes real/fake
images. The source dataset's unused outer transform is not applied. No new
channel swap, horizontal flip or landmark reordering is introduced.

Native collation concatenates all REAL views then all FAKE views without
shuffle. SourceB therefore means whole model2B; prescribed order is validated
before optimization. A reusable PairedImageTrainer handles separate source/view
provenance, actual-model-image loss averaging and complete BN batches. FWA
retains its different generated-first/shuffle behavior and original counters.

samples.jsonl counts real parents once. views.jsonl records both ordered model
views. generation.jsonl records parent/sidecar/mask/pixel hashes, per-source RNG
digests and complete batch Python/NumPy RNG. source-selection.json records the
excluded fake parents. The source's zero-mask division still yields NaN;
no epsilon repair is silently introduced.

The native API swallows exceptions and returns(None,None); its dataset then
fails tensorization despite the apparent fallback. The rewrite surfaces this
same fatal boundary with the original cause, parent and RNG in
generation-failure.json. It never invents a real/black replacement. Partial
receipts remain, including when a spawned worker returns the failure. Removing
source global logging suppression is an explicit infrastructure change.

Training, state and validation
--------------------------------

Defaults retain Adam2e-4/decay5e-4,51 sweeps and StepLR3/gamma0.4. The checked
non-SWA source steps the scheduler once AFTER the entire fit, so the first full
fit does not decay the learning rate. No silent per-epoch repair is made.
Best/last snapshots contain pre-postfit scheduler state; the final task retains
the postfit carrier. Strict checkpoints include the fixed construction/pair
recipe, full model/BN/head, preprocessing, initialization identity and safe
scheduler dictionary. Exact optimizer/RNG resume is not exposed.

Native audits compare all eight68/81-point hull cases,20 stochastic generation
seeds, synchronized outputs, ordered pairs and complete final RNG exactly.
Constructed and actual B4-weight audits at RGB256/source2/effective4 compare
initialization, features/logits/scores, all-three-logit CE, all418 parameter
and image gradients, drop-connect Torch RNG, BN/model state, named Adam update,
scheduler and restored scores byte-for-byte. The head is seeded and untrained;
actual ImageNet initialization is not a historical trained SBI checkpoint.

Tests cover real filtering, pinned sidecars, full pair order, third-class score/
gradient, a manual effective-batch BN/Adam comparison, failure receipts, strict
restoration, real-only training with binary validation and native model fitting/
earlier-best selection/evaluation. Actual spawn workers test both success and
failure transport. An initial cached local downscale class failed pickling;
its replacement is importable and preserves all numeric/RNG comparisons.
Injected validation0.9/0.8 is control-flow evidence, not measured accuracy.

Native batch24/effective48, train frames8 and51 sweeps remain in the example;
bounded audits use source2/effective4 without splitting. Remake manifests and
source-validation selection differ from the full original protocol. All model
execution stays under3GiB RAM, swap0, CPU2 and hidden GPU, with4GiB host reserves.
Raw states, inputs/landmarks, gradients, optimizer, code snapshots, failures and
reproducible commands are linked from configs/migrations/validation.json.

The final complete CPU suite passes189 tests in28 serial capped groups with
zero swap/OOM/GPU, including ten SBI tests and the FWA/Effluxlet regressions.
All discovered test files and tested source/fixture hashes are recorded. Earlier
gate-refused, interrupted and Effluxlet OOM attempts remain separate raw evidence.

Attribute hull variants and unused source helper paths are unconnected.
Historical trained SBI scores, GPU numerical alignment, exact training resume
and full dataset reproduction remain unverified. Lightning dual-expert fusion
and unified partial_fit/remove_batch entry points remain unconnected.
