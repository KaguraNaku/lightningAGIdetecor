ProDet preparation
==================

``ProDetDetector`` composes an injected Backbone with the native1792-to512
projection,three binary experts,feature attention,three stochastic drift blocks,
adjacent-domain Beta mixing andthe final linear binary classifier. The selected
recipe is ``linear_init=False``,``cat_manner=linear``,drift/mixing enabled and
``Op_start=0``. All constructor draws andparameter ordering match the source.

The ordinary ``forward`` returns inference logits. Training calls
``forward_outputs(...,inference=False,epoch=...)`` with the zero-based source
epoch explicitly supplied by the caller. ``loss_terms`` keeps original int64
orFP32 domain labels andmixed confidences. Its three soft cross-entropies,three
drift MSEs andfinal soft cross-entropy retain the source order,in-place overall
alias andgradient paths through the drift targets. The native CPU normal/Beta
draws remain CPU draws;CUDA copies alone are ported to the operand device.

Data recipes require explicit wiring
-----------------------------------

The actual local ``train.py`` does not recognize ``dataset_type=sbiplus_v2``.
Its executed selector chooses the generic base dataset andbatch24 binary images.
The model still divides this batch into four segments;this does not establish
the intended real/blend/BI/deepfake data semantics.

The separately defined ``SBIPlusV2Dataset`` instead collates24 quadruples into
96 images inreal/blend/BI/deepfake order,withoriginal float labels0/1/3/2.
Its original collation was executed ondeclared synthetic marker planes. The
selected whole-face SBI/soft-BI synthesis and equal, ordered quadruple collation
are now independently implemented in ``generation.py``. ``ProDetSourcePlan``
now prepares filesystem source identities, pairing, hashed landmark/neighbor
bindings and complete-source split checks. ``ProDetDataset`` now composes that
plan with native image/NPY decoding and generation through the shared loader.
The unified Task and dedicated Trainer are now prepared, with native PNG
inference/conversion/evaluation verified; actual full-CNN fitting remains pending.
Keep these recipes distinct;
do not flatten specialized domain
labels to binary loss targets orsilently describe a wiring repair as untouched
``train.py`` behavior. The source epoch loop is0..20 inclusive21 epochs.

Verified mathematical scope
---------------------------

* Post-B4 supplied1792x8x8 feature maps atB24->42 andB96->168 match the original
  complete component outputs,all losses,all70 parameter gradients,the input
  feature gradient,one native Adam step andCPU RNG. This excludes the CNN.
* Three full ImageNet-B4 plus randomProDet constructors match782 states,488
  parameter flags/order andCPU RNG. The actual local ImageNet asset is hashed;
  its fresh native unpadded stem andunused randomlast_layer are retained.
* Full32x3x256x256 evaluation matches B4/projection intermediates,all outputs,
  unchanged model states andCPU RNG. The inputs are declared synthetic normalized
  tensors,not evidence of native dataset preprocessing orhistorical scores.
* The unchanged 41 related CPU contracts and ten new Task/Trainer contracts
  pass. The initial combined 51-test attempt had one test Runtime fixture error;
  all ten new tests passed after correcting its device type. This is a checked
  union of results, not a second successful complete 51-test run. Native
  numerical comparisons remain separate.

Specialized synthesis scope
--------------------------

The actual original ``SBIPlusV2Dataset.__getitem__`` and ``collate_fn`` were
executed for three seeds, each with a full batch of 24 quadruples. All 288 output
RGB256 images, SBI/BI masks, float domain labels, nearest-foreground choices and
Python/NumPy/Torch RNG states match exactly. Both raw native/candidate batches,
declared inputs, source/config snapshots and resource receipts are retained.
These are declared fixture bindings, not a filesystem or full-dataset test.

BI foreground selection runs after SBI synthesis through an explicit callback;
moving the random nearest-face choice before SBI would change its pixels. Only
the selected whole-face, soft-BI, top-three, equal/full sequence recipe is
connected here. The synchronized four-image augmentation and half-mean/std
normalization preserve the source recipe. ``prodet_labels`` retains float
0/1/3/2 targets; a separate binary ``labels`` view is provided for coverage.
The collator carries per-view identities, survives spawn serialization, keeps
final partial quadruple batches and refuses discarded or invalid sources.

``select_foreground`` consumes an already prepared ordered neighbor table. It
does not load the original pickle sidecars or reproduce ``init_nearest``'s
mutation-while-iterating filter. ``ProDetSourcePlan`` below provides that
explicit binding/filter protocol. Nonfinite/empty BI color-transfer results fail visibly;
there is no added epsilon, replacement source or silently repaired pixel rule.

The data differential and reproducible commands are indexed in
``experiments/detector_rewrite/prodet_data_audit_20261104.json``.

Source binding and pairing preparation
-------------------------------------

``protocol.py`` requires the exact complete train manifest, including sources
excluded from sampling. Each image gets a byte hash; real sources require an
explicit landmark path/hash and 256-coordinate declaration. Explicit original
Windows path strings retain the native grouping and lexicographic ordering;
actual files remain bound to relative POSIX paths inside the dataset root.
Manifest order is declared input, not a reproduction of upstream corpus discovery
or its shuffle RNG. The source's mandatory fake-group281 exclusion, 32 slots per
group, real-slot modulo8 and short-list last-frame repetition are retained.

The compiled JSON neighbor table binds the complete manifest hash/member order
and keeps all original raw ordered edges. The original mutation-while-iterating
filter is applied literally, including surviving same-video entries. No nearest
search, deduplication or tail filtering is introduced. Every raw key/edge, even
beyond the selected first three, must resolve to a real train member. An invalid
or emptied table fails explicitly. Train/validation checks cover original frame
IDs, paths and videos, including group281 and unused foreground sources; they
do not claim subject or filesystem-alias separation.

An original-source differential executes grouping, indexing, length, the intact
getitem path-selection prefix, actual pickle-based ``init_nearest`` and foreground
choice. Across 48 declared byte-fixture pools, all 3,072 pairs, 12,288 view bindings
and 2,265 foreground choices/Python RNG states match. The 535 surviving same-video
edges are retained as source behavior. These fixtures are not decoded images or
landmarks. The first audit run failed only at its final source-hash check because
it passed a string to a Path-only helper; both runs and snapshots are retained.

The implicit original pickle files were absent at the checked repository-root,
``training`` and ``training/dataset`` locations. No author neighbor corpus was
invented. Actual author asset acquisition/conversion and Task wiring remain
pending. The Dataset now validates shape/finite numeric NPY arrays and supports
worker synthesis on explicitly bound filesystem inputs. The original unused
``landmark_dict`` pickle read is not reproduced in production.

Commands, raw fixture pools, native prefix/filter outputs, test receipts and the
negative run are indexed in
``experiments/detector_rewrite/prodet_protocol_audit_20261104.json``.

Filesystem Dataset and worker preparation
----------------------------------------

``ProDetDataset`` binds the complete source plan, validates every real NPY file
(including unused/excluded sources), reads native OpenCV BGR-to-RGB/cubic256
images and casts unscaled NPY coordinates to int32. It repeats the background
landmark read after the BI foreground selection, preserving source IO order.
Each requested quadruple retains source bindings, selected foreground, raw
Python/NumPy RNG, mask/view fingerprints and explicit short-list clamp status.
``ProDetDataCollator`` carries those receipts and domain-ordered view identities
through the existing loader; float domain targets stay separate from binary
coverage labels. Decode/hash/control-file failures keep the requested identity
and fail after one attempt. No recursive or neighboring fallback is introduced.

The original specialized constructor/getitem/collator and original filesystem
decoder plus original torchvision ToTensor/Normalize definitions were executed
on declared PNG/NPY fixtures. Windows identity strings were explicitly mapped to
their POSIX files, and upstream discovery was replaced with the declared pool.
A seeded shuffled epoch matches all 64 quadruples/256 output views in full
24,24 and final16 quadruple batches: pixels, masks, float targets, sample indices,
foreground choices, constructor RNG and per-batch Python/NumPy/Torch RNG are
exact. Source lists include native clamping and the excluded281 group. Both raw
epochs, varied input PNG sizes, float64+.25/int32 NPY files and config/source
snapshots are retained. This is fixture data, not an author corpus or score test.

Seven Dataset contracts verify complete view coverage, repeated seeded spawn
loading, malformed unused NPY files, mutated bindings, undecodable requested
images, collator identities and fatal worker-to-parent error propagation. The
one-worker workflow checks conditional replay under remake worker seeding;
it does not reproduce original upstream or eight-worker global RNG. The initial
native harness failed on a missing ``Tensor`` annotation binding before data
iteration. Only that reference namespace was corrected; the failed run remains.

Commands, raw native/candidate epochs, generation/selection receipts, resource
records and negative run are indexed in
``experiments/detector_rewrite/prodet_dataset_audit_20261104.json``.

Strict complete checkpoint transport
------------------------------------

``checkpoint.py`` requires all 782 native Detector entries with exact CPU dtypes,
shapes and native strides, including BN counters and the unused backbone head.
Partial backbones, missing buffers, mixed prefixes, sparse/meta tensors,
nonfinite values and implicit dtype/layout repair are rejected. Bare states,
original Trainer ``state_dict`` envelopes and uniform DDP prefixes use the
existing safe weights-only reader. Original epoch/optimizer/scheduler metadata
is retained in the source file and is not resumed by this weights importer.

Callers must explicitly supply ``selected_recipe()``: the linear construction,
drift/mixing switches and zero-based operation-start epoch are nonpersistent
and cannot be inferred from tensor shapes. This API transports weights under
that declared recipe; it does not establish how a historical model was trained.
The Task layer validates its saved construction/preprocessing metadata separately.
Restoration uses a meta constructor plus independent validated CPU clones,
preserving all 488 trainable parameters, native order, input isolation and CPU
RNG. It loads no ImageNet asset and does not fall back to random parameters.

The retained actual-source full native constructed checkpoint was imported,
freshly serialized to a new complete state file, then reloaded. Both whole
32x3x256x256 FP32/default-MKLDNN evaluations match all original seven outputs,
B4/projection sites, all 782 unchanged states, parameter flags/order and RNG.
The new ZIP serialization was actually exercised; no older archive substitutes
for that write. Inputs are declared normalized synthetic tensors and weights
are constructed ImageNet-B4 plus random ProDet heads. This is not historical
trained-score alignment, a Task conversion/CLI test, CNN backward or a full run.

The six full-geometry contracts retain their raw format/negative-case files and
verify strict import/restore, isolated storage and nonpersistent recipe guards.
No numerical worker failed in this stage. A prelaunch hook-closure lint warning
was fixed before launch and recorded separately. Raw checkpoints, commands,
reference lineage, outputs and capped resource receipts are indexed in
``experiments/detector_rewrite/prodet_checkpoint_audit_20261104.json``.

Unified Task and Trainer preparation
-----------------------------------

``ProDetTask`` keeps ``ExperimentTask -> Detector -> Backbone`` and requires an
explicit ``native_entrypoint`` or ``specialized_quadruples`` profile. It saves
typed construction, preprocessing roles, probability semantics and the optional
neighbor identity. Restoring complete Task weights needs no old initialization
asset; conflicting recipe/preprocessing/profile metadata fails. The specialized
training sidecar is opened only when preparing training data. The preset
``configs/experiments/prodet_ffpp.yaml`` declares the intended quadruple wiring
repair and placeholders for genuine source assets.

``ProDetImageDataset`` owns filesystem OpenCV decoding for the generic entrypoint,
validation and evaluation. The original decoder/getitem/augmentation/torchvision
and collator were executed against Task preprocessing for three seeds and three
roles, totaling288 declared PNG views. All pixels, labels and Python/NumPy/Torch
RNG match. Validation uses original test-mode clean pixels as formal testing does;
the remake source-val split is an explicit selection policy. Fixtures reuse the
same files between transform roles and do not constitute an isolated training
experiment. Native parquet pixel parity is not established by this PNG audit.

An actual original full ProDet model at its constructed native checkpoint was
evaluated on32 of those PNG views. Its probabilities match the Task, a freshly
converted/restored engine Task envelope and a separate ``engine.py evaluate``
process exactly. Image32/video4 metrics match independent sklearn calculations
and all six CSV/JSON cells. These fixture labels and constructed heads are not
historical trained-score or full-corpus evidence.

``ProDetTrainer`` preserves native zero-based epochs (default0..20 inclusive21),
Adam parameter order and complete batch objectives. Specialized float0/1/3/2
targets stay separate from binary coverage labels; formal generic entrypoint
targets remain int64 binary. Generated-source receipts and exact saved-tensor
transport are retained per batch. CPU fitting requires fp32, batch24 (96 images
for specialized data), validation32, final partial batches and adequate persistent
storage for the entire retained run plus8GiB floor and4GiB checkpoint headroom.
The capacity estimate is based on the full graph's logical saved bytes with25%
archive allowance; it is not a measured RAM/disk peak. Insufficient space writes
a negative preflight receipt before model movement/optimizer/training. GPU/AMP
training and exact optimizer/RNG resume remain unsupported pending validation.

Ten contracts cover metadata/readiness, both data profiles, nonpersistent epoch,
unmodified float targets, full488-parameter Adam order and a preflight proving
training does not start with insufficient storage. Loss-route/epoch controls do
not substitute for numerical full-B96 CNN gradients. The first combined test
attempt and initial Task-audit path-type error are retained with corrected runs.
Raw inputs/outputs/configs, commands and exact validation scopes are indexed in
``experiments/detector_rewrite/prodet_task_audit_20261104.json``.

The initial full-constructor harness failed onlist-versus-tuple stride reprs.
The correction canonicalizes only that representation andchecks every numeric
convolution attribute;the source failure andraw checkpoint remain preserved.
Readonly archive reuse occurs only after full typed-value/hash verification.
It does not claim a new ZIP serialization was exercised onthe reuse path.

Pending integration
-------------------

The unified Task/Trainer/preset are prepared and the inference/conversion/CLI
workflow is verified on declared native PNG fixtures. The method remains pending
the actual full native fit/gradient and complete-regression gates. Detector
weight import and Task state restore do not resume optimizer/RNG training state.
The Dataset requires explicit real author source/neighbor assets before an
author-data run. Actual full-CNN gradients at the selected96-image specialized
geometry,the fitted restore/evaluate workflow andthe complete449 CPU suite remain pending.
The last complete386-test pass predates these additions. No full experiment,
historical trained detector orGPU AMP reproduction is claimed.

The full candidate graph's logical unique saved-storage estimates are
7,316,105,712bytes forB24 and29,262,895,872bytes forB96. These are not measured
RAM ordisk budgets. B24 is not substituted for the missing specializedB96 gate.
The two legacy unused ``backbone.last_layer`` parameters stay in the model,
constructor RNG andoptimizer ordering even though features-only training gives
them no gradient. Resource preflights retain8GiB reserves;no batch orprecision
reduction is used to obtain alignment.

Original run IDs,raw tensors,commands,environment,snapshots,failures andresource
receipts are indexed in
``experiments/detector_rewrite/prodet_preparation_audit_20261104.json``.
