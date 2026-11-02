PCL-Xception and I2G on the unified experiment pipeline
========================================================

PCLTask composes the independent SLADD-Xception/PCL model, a pinned ordered
neighbor bank, real-only I2G dataset and whole-pair trainer. Training, source
validation, selected checkpoint restoration and final image/video evaluation
use the existing engine. Production code never imports DeepfakeBench.

The source algorithm is preserved for successful inputs. Remake uses explicit
train manifests and stable first-occurrence deduplication, with source-domain
validation for checkpoint selection. These transport/selection choices differ
from the original full experiment. Historical trained PCL scores, GPU alignment
and full151-sweep dataset reproduction remain unverified.

Native model semantics
------------------------

RGB256 produces final2048x8x8 features and a block7-side728x16x16 feature. The
SLADD backbone keeps all its fresh type/magnitude/regression heads, even though
PCL does not use them. All blocks use non-inplace activations and the native
residual evaluation order. Ordinary backbone forward exposes classification;
the auxiliary SLADD forward/training route is not connected by this migration.

Correlation calls theta TWICE, retains unused trainable phi, divides the spatial
Gram by sqrt728, transposes/contiguously stores it, applies sigmoid and reshapes
to Bx16x16x16x16. There are166 parameter tensors;158 receive gradients. The
backbone's six extra-head tensors and phi's two tensors remain without gradients.
No unused module is dropped or used to replace the second theta call.

Ground-truth masks use UpsamplingBilinear2d(16/256), including align_cornersTrue.
The native Masks4D loop stores the RAW mask in the first width slot of EVERY
outer row. Only later width slots use1-abs(a-b). The independent broadcast
implementation retains that unusual first column exactly. An all-zero16x16
mask therefore has target mean0.9375 rather than1. Loss is CE plus weight1 BCE
of sigmoid correlation and the native target. Inference skips correlation and
needs no mask. No BCEWithLogits or uniform-affinity correction is introduced.

The exact original Xception ImageNet file initializes234 keys, while classifier,
type/magnitude and map heads keep source constructor values. Pointwise matrices
are expanded as in the native importer; missing BN counters retain native zero
initialization. The explicit importer rejects an unrelated/incomplete trunk.
Full native actual-I2G source2/effective4 model audits agree byte-for-byte for
constructor RNG, initial state, every output/target/loss, parameter/image
gradient, BN state, complete Adam update and strict model-restored scores.
These use pretrained vision plus seeded untrained heads, not a historical
trained PCL detector or full-dataset experiment.

Pure I2G generation
---------------------

The background landmarks are clipped0..255 and their13 forehead entries are
reordered before optional68-point truncation and four native hull choices.
Foreground images are warped with affine and elastic masks; the elastic IMAGE
output remains discarded. Native color transfer, blend-mask arithmetic and the
randint3 draw over ratios[1,1,1] remain. Source photometric augmentation and hflip
helpers are dormant and are not activated. Shared photo transforms apply to
real/fake pixels without the FaceXray channel swap. Raw blend masks remain
float32 and are not replaced by FaceXray's4*alpha*(1-alpha) boundary.

Collation is all REAL then all FAKE without shuffle, with masks/labels aligned.
Twenty preserved native seeds/40 source pairs agree exactly for pixels, raw
masks, downsampled masks,4D targets, labels/order and final Python/NumPy/Torch RNG.
A separate actual-base-NPY-loader audit checks the complete dataset/collator
for40 seeds (20 float32,20 float64), first-source fallback and missing-sidecar
boundaries, including exact resulting RNG. Raw success and failure artifacts
remain available.

Data binding and failure boundaries
-------------------------------------

I2G selects real backgrounds and deduplicates them using list(set(...)); this
legacy hash-dependent inventory order must be distinguished from remake input
selection. The published nearest-face table selects a foreground or falls back
to self. The published landmark dictionary is loaded by the legacy initializer,
but actual numerical landmarks come from per-frame NPY sidecars via the base
loader. Foreground landmarks are also read even though they are unused. Do not
substitute FaceXray's dictionary coordinates for the sidecar values or silently
convert their dtype. The native base loader returns float64 zero81x2 for missing
sidecars. An actual-base-loader probe preserves20 float64 near-integer seeds;
the independent generator matches every pixel/mask and RNG without casting.
Missing foreground sidecars leave the pair and RNG unchanged. Missing background
sidecars produced4 finite and16 nonfinite masks across20 seeds; retain this
negative evidence instead of treating native zero fill as automatic rejection.
The dataset records missing sidecars and stops explicitly on nonfinite
generation rather than train on NaNs. Finite native zero-fill outputs remain.

On image decode failure, native __getitem__ recursively returns index0. A
successful first-source replacement and its extra RNG are preserved in the
native probe. If index0 also fails, recursion ends in RecursionError; the probe
used an explicit recursion100 cap inside its cgroup. The adapter retries index0 once, preserving successful replacement pixels and
RNG. If that replacement also fails it saves a diagnostic and stops. This
changes the persistent-failure boundary explicitly. Requested slot IDs remain
unique while every view also records its actual pixel background. Per-epoch
coverage reports both requested slots and unique actual sources/occurrences.

The Task/Trainer retain native151 default inclusive sweeps, Adam2e-4/decay5e-4,
full paired BN batches and raw-mask supervision. Non-SWA StepLR3/gamma0.4 runs
once AFTER the complete fit, as in the source. Epoch snapshots retain pre-fit-
completion scheduler state; the final Task carries its post-fit state. Exact
optimizer/RNG training resume is not supported. Source-domain validation and
optional early stopping are remake selection policies, not original reproduction.

Prepare the neighbor bank offline with:

::

  python3 tools/bounded_run.py --output NEW_ABSOLUTE_RUN --memory-mib 1536 --timeout 600 -- .venv/bin/python -m detectors.pcl_xception.prepare --manifest /absolute/train.jsonl --neighbors-pickle /absolute/trusted-nearest-face-info.pkl --output /absolute/NEW-bank

Every manifest member needs metadata.pcl_bank_id matching its published
landmark-path key. All ordered neighbor references must bind the SAME complete
train inventory; duplicates and absent-key self fallback retain native RNG.
There is no silent neighbor truncation/filtering or replacement with a landmark
dictionary. Neighbor arrays are memory-mapped and fingerprinted. Coordinates
come from raw per-frame NPY sidecars: native frames-to-landmarks derivation, or
metadata.landmarks={path,coordinate_size:256,sha256:optional}. Present files and
missing-file absence are pinned before workers start; no dtype cast/rescaling.
Training backgrounds filter real0 and deduplicate storage identities before
sampling. Foreground inventory, including unused fake rows, is checked for
validation overlap. Inference restore needs neither the bank nor NPY files.

configs/experiments/pcl_xception_ffpp.yaml preserves source batch32/effective64;
our bounded CPU workflow checks use a separate declared miniature source2/
effective4 fixture. Smaller CPU checks do not prove full-batch/full-data training.

Evidence is under persistent /var/lib/codex-experiments-naku. Native I2G data
probe-v2 preserves complete20-seed tensors/RNG and fallback logs. Initial oracle
namespace failure omitted random; that failure/source snapshot is retained,
and the corrected isolated loader changes no native arithmetic. Run
experiments.detector_rewrite.pcl_check reference then candidate in one fresh
absolute case with distinct bounded run dirs; use --data pointing to a pinned
native seed artifact and optional --weights for actual ImageNet initialization.
experiments.detector_rewrite.pcl_components replays all native data seeds in a
fresh case. Model runs use3GiB RAM, swap0, CPU2 and hidden GPU; data probes use
1536MiB. Historical trained scores, GPU alignment, exact optimizer/RNG resume
and full dataset reproduction remain unverified. Existing Lightning dual-expert
fusion and unified partial_fit/remove_batch remain unconnected.
