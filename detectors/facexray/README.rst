FaceXray generated pairs and raw boundary supervision
======================================================

FaceXrayTask composes a mapped FaceBank, FaceXrayDataset/PairCollator, method
Trainer, raw-boundary Detector and lazy independent HRNetW48. The engine only
uses its optional dataset hook before sampling/workers. Validation and testing
retain standard RGB256 decoding with native mean/std0.5. Production imports no
DeepfakeBench, torchvision or dlib. HRNet follows the Microsoft variant credited
to Bin Xiao and Ke Sun in the pinned native source.

Literal model and initialization
---------------------------------

The actual source hardcodes HRNetW48 despite the YAML B4 field. It concatenates
720-channel64x64 features, predicts an unbounded256x256 boundary, pools that mask
to128x128 and applies FC1024/BN,128/BN and two logits. HRNet's default bilinear
align_cornersFalse and decoder align_cornersTrue remain distinct. All1001
parameter tensors remain trainable;66 unused backbone head tensors retain
constructor RNG/state without gradients.

The exact author-distributed ImageNet checkpoint loads1954 tensor keys. Its
classifier.weight/bias are ignored, while the unused fc.weight/bias keep fresh
constructor values. They are not renamed. The explicit importer checks this
complete layout and rejects wrong trunks instead of silently initializing
random weights. Constructed studies must explicitly select initialization:random.

Scoring uses softmax[:,1]. Training always adds1000 times raw-boundary MSE to
binary CE after the source float32 target/prediction casts. There is no sigmoid/
BCE substitution. Original views have zero boundaries even for fake label1
backgrounds. Defaults retain eleven sweeps, Adam2e-4/decay5e-4 and no scheduler.
Native BN sees the whole model2B; microbatching is rejected.

Explicit supervision transport
--------------------------------

Published banks contain159199 finite float32(81,2) landmark arrays and31935
real-background neighbor entries of100 candidates. All references resolve;
one self edge and69 arrays outside0..255 are retained. Landmark backgrounds
include31935 real and127264 fake frames. Native FFBlend iterates ALL landmark
keys and uses self foreground for a missing neighbor entry. SBI's real-only
filter would change this algorithm and is not reused.

A compiled bank.json binds the exact train manifest hash and every member ID.
Every foreground belongs to that SAME complete inventory, so normal train/val
image/video-disjoint checks also cover auxiliary exposure. Each source record
supplies metadata.facexray_bank_id matching its trusted legacy key. Manifest
labels explicitly replace native path-component6 guessing.

Invoke the offline preparer through bounded_run.py with fresh absolute outputs:

  .venv/bin/python -m detectors.facexray.prepare --manifest TRAIN_MANIFEST
    --landmarks-pickle LANDMARKS_PKL --neighbors-pickle NEIGHBORS_PKL
    --neighbor-policy provided --output NEW_BANK_DIRECTORY

Only this explicitly invoked import unpickles trusted legacy assets. Runtime
uses pinned JSON and read-only mapped NPY arrays. Provided lists remain complete,
ordered and unchanged. Unbound foregrounds, empty present entries, changed arrays
and wrong manifests fail. Neighbors are never silently filtered/deduplicated;
coordinates are neither rescaled nor clamped.

A published full FF++ bank can conflict with a held-out remake split. An explicit
alternative uses --neighbor-policy training_kdtree --neighbors 100, without
--neighbors-pickle. It applies native KDTree query/order arithmetic to manifest
real0 members while retaining all mixed-label backgrounds. The first query
result is removed, including native tie/self-edge behavior. This deliberate
train-scoped input-bank/protocol change is recorded, not presented as the
published full-bank experiment. At least101 real members are needed for100
neighbors. Small fixtures explicitly use provided miniature lists or k1.

Generation, sequencing and errors
-----------------------------------

Generation belongs to dataset __getitem__: each foreground selection and full
augmentation precede the next source selection, including spawn workers.
Foreground augmentation also precedes a lazy background decode. Source/background
augmentations run unconditionally. Five hulls, a second background-mask
deformation, five-point affine versus first48 Delaunay warp, native clipped
interpolation, color transfer and feather blending remain. Zero variance and
empty-mask behavior has no epsilon repair.

Postprocessing preserves the channel swap and computes/discards another
augmentation output while retaining its RNG. Boundary targets use
4*blurred_alpha*(1-alpha) in float64, cast to float32 only in the native loss.
Fake-all then real-all views are jointly shuffled with masks/labels/parents.
Before any forward/update the trainer verifies boundary hashes and zero original
boundaries. No replacement image or relabeling is introduced.

samples.jsonl counts background parents; views.jsonl counts whole model2B.
generation.jsonl preserves foreground/background IDs and labels, self fallback,
geometry/pixel/boundary hashes and batch Python/NumPy RNG. source-selection.json
binds bank/manifest identity and mixed-label counts. loss-components.json retains
image-weighted CE, raw MSE and total means. Fatal failures preserve original
cause, parent, bank and RNG in generation-failure.json, including worker errors;
partial source/view receipts remain.

State, evidence and limits
----------------------------

Strict raw imports and task state retain full model/BN/boundary/head tensors,
fixed construction/pair recipe, preprocessing, source/init identities and bank
receipt. Inference restoration uses meta construction and opens no training
bank or legacy weights. Future fitting can relocate a bank only with identical
header/content identity. Exact optimizer/RNG training resume is not exposed.

Constructed and exact author-ImageNet native RGB256/modelbatch2 audits agree
byte-for-byte for constructor RNG, initial state, all outputs/losses,935 parameter
and image gradients, BN/model state, complete Adam update and restored scores.
The decoder/classifier are seeded and untrained, not historical trained weights.
Data audits compare20 seeds/40 parents across affine/triangular paths, mixed
labels, self fallback, every paired image/boundary/label/order and final RNG.
The full compiled-bank/manifest dataset/collator is included in that data audit.

Eleven CPU tests cover model/loss contracts, pinned native golden pixels/RNG,
mapped bank identity/ranges, explicit preparation/tie behavior, manual whole-
batch BN/Adam, corruption before forward, actual spawn success/failure, strict
conversion and native learning/best-state selection/evaluation. Injected val
0.9/0.8 proves control flow only. Workflow inputs/configs/banks/checkpoints are
preserved under persistent experiment directories. All model execution uses at
most3GiB RAM, swap0, CPU2 and hidden GPU; data-only audits use tighter caps.

The example retains source16/effective32 and32 frames/video. Bounded native
learning uses source1/effective2 without splitting BN batches. Remake manifests,
validation selection and any explicit train-scoped bank differ from the original
full FFBlend protocol. Historical scores, GPU alignment and full dataset
reproduction remain unverified. Lightning dual-expert fusion and unified
partial_fit/remove_batch entry points remain unconnected.
