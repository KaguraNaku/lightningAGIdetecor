UCF common and specific forgery detection
==========================================

UCFTask composes a method-owned paired dataset, whole-batch Trainer and an
independent common/specific reconstruction model through Task -> Detector ->
Backbone. Production imports no DeepfakeBench. Local original source hashes,
runtime references and gate scopes are recorded in the migration ledgers.

Native mathematics
--------------------

Two trainable Xceptions use adjust_channel512 features and retain their unused
2048->2 classifiers. This differs from IID's512-input classifier and must retain
its original constructor RNG. Only the two stateless convolution/block building
classes are reused from the verified IID backbone module; no recognition network
or optional identity dependency is imported. Each trunk may load actual local
ImageNet weights. Fresh adjustment layers/heads retain the source seeded draws.
Missing/wrong initialization fails explicitly rather than the source's silent
random fallback. Explicit initialization: random is a separate declared choice.

Separate1x1 projection blocks produce256-channel common/specific features. The
specific head has len(train_dataset)+1 classes, default5; specific_classes may
explicitly select another count with its matching label map and strict state.
Shared binary logits yield softmax class1 scores. Native head dropout occurs
AFTER its final linear, including both inference heads when model.train() is set.
Both encoders and both heads execute in inference; only reconstruction is skipped.

Training performs a Python-random .7 in-place permutation within real half then
fake half. Common/specific concatenation and content chunks feed ONE shared
ConditionalUNet four times: self(fake), self(real), fake-style/real-content,
real-style/fake-content. AdaIN uses native unbiased variance and eps1e-5. Stage
convolutions are shared across style and content in native order; dropout.3,
bilinear align_cornersTrue and final tanh remain. The active double convolutions
do not use the source's dormant spectral-normalization helper.

Loss is common CE +.1specific CE +.3sum(four mean L1) +.05contrastive. The four
reconstruction terms retain self-fake,self-real,cross-fake,cross-real order.
Contrastive uses margin3, native default pairwise-distance epsilon, Python-shuffled
real/fake common anchors and per-specific-label shuffle/pop anchors in insertion
order. Its four terms keep native addition order. No loss reweighting, detach,
normalization correction, microbatching or class-collapsing repair is introduced.
All354 parameter tensors survive;350 have gradients. Four unused classifier
tensors remain trainable and in the optimizer group but have no Adam state.

Paired transport and provenance
---------------------------------

All fake manifest rows form the indexed training view, preserving manifest order.
For each fake, random.randint samples one real with replacement BEFORE decoder
IO. Decode fake then real, independently augment fake then real, and normalize
RGB256 with mean/std.5. Collation yields ALL real images then ALL fake images,
with binary0/1 and separate native specific labels. Effective batch size is twice
the loader's source-pair batch. Last batches remain whole and are not regrouped.

Real specific labels are0; fake labels must be1..specific_classes-1. Explicit
metadata.specific_label takes precedence and its binding is recorded. Otherwise
the declared specific_label_map binds category strings; the default canonical
FF-DF/F2F/FS/NT mapping is1/2/3/4. Native newer *_fake aliases map to1 in its
config, whereas canonical types differ. Unknown aliases fail unless explicitly
declared; they are never silently corrected or inferred from binary labels.
Explicit metadata can declare another reference mapping without renumbering.

Training receipts separate unique indexed fake coverage, the full real pool,
actual sampled real occurrences and unique real observations. Duplicate sampled
reals remain distinct model images. Validation overlap is checked against the
ENTIRE source inventory, including real rows outside the fake-indexed view.
Source failures stop with requested fake, already-sampled real and RNG receipts;
the native pair body has no recursive source replacement. No skipped/relabelled
source or generated surrogate is introduced. Spawned workers preserve receipts.

Parquet transport uses explicit manifest records and specific metadata; this
replaces native pairDataset's unsupported tuple-path .replace call deliberately.
Decoded uint8 pixels follow the same cubic256/augmentation/normalization path,
checked against image-file transport. It does not claim that native tuple-path
pairing ran. Ordinary validation/evaluation is unpaired and requires no specific
labels. Optional native label/probability observers are retained in the direct
model API; the unified evaluator owns actual image/video metrics.

Verification and protocol boundaries
--------------------------------------

The isolated native and independent model/Task audits compare constructor tensors,
modes/trainability and Python/NumPy/Torch RNG; every inference/training output,
loss,350 parameter and image gradients; full BN/model/Adam updates; and strict
restored scores byte-for-byte. Constructor seed1024 and explicit training Python
seed1 exercise the original in-place shuffle at full256/effective4 geometry.
The original reference uses actual ImageNet trunks and fresh untrained heads.

Twenty actual native pairDataset seeds compare complete pixels, binary/specific
labels, collation order, actual sampled real provenance and all RNG. Discovery is
replaced by an explicit miniature inventory in that audit, not claimed equivalent
at the full dataset level. Original legacy-weight mmap audit failure and the
interrupted initial Task check retain separate raw lineages and terminal evidence.

CPU tests cover known contrastive/AdaIN values, full gradients/inference execution,
native paired goldens, label mapping, full-pool isolation, fatal decode receipts,
spawned workers, explicit Parquet transport, strict raw state and complete native-
resolution best/last selection/save/restore/evaluation. A pinned-AUC fixture tests
selection independently from the separate unmocked miniature validation workflow.

Default source settings are Adam2e-4, decay5e-4, no scheduler and six inclusive
sweeps. Remake source validation/selection is an explicit policy. Example source
pairbatch16/effective32 is not reproduced by bounded miniature pairbatch1/2 tests.
Standalone inference needs neither initializer files nor training data; strict
state includes both encoders, unused classifiers, projections, both heads and
the complete reconstruction branch. Exact optimizer/RNG training resume,
historical trained scores, GPU alignment and full dataset experiments remain
unverified. Lightning dual experts and unified partial_fit/remove_batch remain
unconnected.
