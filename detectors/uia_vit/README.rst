Independent native UIA-ViT rewrite
==================================

UIAViTTask composes a full native ViT-B/16, labeled patch-bank collection,
LedoitWolf fitting and CE/PCL training. UIATrainer uses shared whole-batch
fitting, receipts, source validation and best selection. A lazy backbone entry
loads the independent implementation without timm or old detector imports.

Native model and initialization
--------------------------------

The preset is RGB224, patch16, dimension768, depth12, heads12 and MLP ratio4,
with qkv bias and LayerNorm eps1e-6. The classifier consumes concatenated CLS
and weighted patch vectors (1536 inputs). Intermediate patches come from
norm_middle before block6; attention indices8..12 collect preceding-block raw
qk logits, not softmax attention. Progressive localization is uniform before
step1/8, then blends uniform and sigmoid localization using k12/threshold0.7.
Evaluation uses step1. No smaller backbone or reduced geometry is substituted.

The checked local timm initializes truncated normals by inverse CDF. Current
PyTorch uses rejection sampling, changing seeded weights and random draws.
The rewrite explicitly retains the native std0.02/bounds[-2,2] inverse-CDF
recipe. The initial candidate mismatch and corrected lineage replay are kept.
ImageNet import discards only the old head; new middle norm, learned consistency
scalars and 1536-input binary head retain native initialization. Missing files
fail without fallback or network access. The exact referenced ImageNet file,
download receipt and checksum are archived separately from detector checkpoints.

Labeled evaluation and label-free scores
-----------------------------------------

The source wrapper requires labels even during evaluation, appends bank features
and increments its counter. The unified evaluator now optionally calls
predict_batch(batch, runtime=...) when an adapter provides it. UIA uses that
route and retains these source state mutations, including collection from
source-validation batches. Ordinary adapters continue to use predict(images).
Labels select bank membership and returned attention subsets; they do not enter
the classification head. This preserves the local algorithm rather than silently
removing its evaluation collection behavior.

UIA's separate label-free predict(images) is read-only evaluation at step1.
It produces the same classification scores without inventing labels or changing
banks/counter. This route does not reproduce the labeled wrapper's side effects.
Both routes and their distinct state behavior are verified explicitly.

Banks, covariance and loss
--------------------------

Each call retains up to four real images' 14x14 patches and four fake images'
inner3:11 patches, exactly as the source selects them. Every selected feature
value and its order is preserved. Float32 chunks are stored in private temporary
directories on disk (default /var/tmp), rather than retained as growing resident
Tensor lists. Fitting maps a lossless concatenation in the same order and still
calls the original sklearn LedoitWolf estimator and Torch mean/casts. No bank
truncation, covariance approximation, epsilon clamp or sample substitution is
introduced. Full estimator temporaries can still be large; model runs must use
bounded_run.py. Empty-class banks fail explicitly instead of manufacturing data.

Training fits statistics before the current forward when the nonzero previous
counter is divisible by batch_per_epoch//2, then clears both banks. Native
batch_per_epoch1796 and nEpochs10 remain explicit; the fit example uses11 sweeps
to match the source runner's inclusive epoch range. Changing loader membership
or evaluation cadence changes bank/counter history and is not full reproduction.

PCL starts after the post-forward counter exceeds one epoch. It retains native
Mahalanobis arithmetic, hard region indices, target-matrix loops and the learned
c_real/c_fake/c_cross aliases. The objective is CE+0.06*PCL plus the native
0.05 reciprocal and0.5 cross-scalar regularizers. Source hardcoded CUDA transfers
are replaced by the actual tensor device. CPU alignment uses a disclosed
allocation/transfer-only reference port; unmodified CPU failure is retained.

Complete checkpoints
---------------------

New checkpoints embed ordered bank tensors, means, precision matrices and counter
alongside the model and construction recipe. Original cache/source files are
not needed. PCL aliases bind to materialized model Parameters after assign-based
restoration. Best selection restores statistics and model together, retaining
later negative epochs separately. Exact optimizer/RNG resume is not claimed.

Raw initial_checkpoint conversion requires an explicit initial_method_state
sidecar (torch.save dictionary) containing batch_cnt, real_feature_list,
fake_feature_list, real_feature_mean, real_inv_covariance, fake_feature_mean and
fake_inv_covariance. Lists are ordered CPU float32 tensors; means/precision are
both None before fitting or tensors of shapes768 and768x768. Lost source state
cannot be inferred from weights, so imports do not silently reset it.

Validation scope
-----------------

Reference and candidate execute sequentially at native B/16 geometry and bounded
batch2. Constructed trajectories exercise warmup, explicitly positioned covariance
and PCL boundaries, progressive localization, labeled evaluation and restoration.
All tensor dtype/shape/full-byte SHA256 fingerprints, gradients, optimizer state,
losses, model weights, banks and fitted statistics agree exactly. Initial/final
source weights, inputs, raw outputs and complete method states are preserved;
fingerprints avoid duplicate large gradient archives. Counter jumps do not imply
1796 prior batches were trained. Actual ImageNet initialization is checked
separately with an untrained seeded classifier.

The example retains source RGB224, mean/std0.5, disabled augmentation, batch64,
Adam lr3e-5/weight_decay1e-5 and no scheduler. Remake manifests and selection
differ from the original full experiment. Historical trained scores, GPU numeric
alignment and full-dataset reproduction remain unverified. See
configs/migrations/validation.json for artifact paths and reproducible commands.
