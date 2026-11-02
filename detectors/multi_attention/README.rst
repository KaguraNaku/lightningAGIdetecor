Independent local MultiAttention rewrite
========================================

MultiAttentionTask composes the independent B4, four attention maps, V1 texture
enhancement, bilinear pooling and a local/global ensemble. MultiAttentionTrainer
owns warmup CE, late AGDA plus CE/RIL, method state and the checked local scheduler
placement. Shared loaders, sample receipts, source validation, model selection
and frame/video evaluation retain the existing pipeline. Production imports no
legacy detector, EfficientNet library or Kornia.

Source behavior retained
------------------------

Construction initializes the backbone, then RIL, then executes a zero-image
feature probe in training mode under no_grad. That probe changes BN statistics
and counters. Later attention/texture/projection initialization retains native
RNG order. Feature extraction follows the source's named_children traversal:
stem/head Swish are not executed because their module is registered after the
pooling stop point. MBConv drop connect is not enabled. Endpoints and all native
parameter/buffer names are preserved.

The phase counter increments on every forward, including inference. Warmup uses
the backbone classifier while count <= backbone_nEpochs * batch_per_epoch. Late
training performs a detached first feature/attention pass, AGDA and a second
train-mode feature pass before the ensemble and loss. Late inference explicitly
skips AGDA; eval() alone does not select that source branch. Global projection
applies Hardswish twice. dropout_final is constructed but unused. These behaviors
are retained rather than silently repaired.

AGDA samples attention indices from sqrt(spatial_sum+1), normalizes by spatial
maximum+1e-8, draws Python threshold/zoom and Torch Gaussian noise, and uses
bilinear align_corners=True resizing plus horizontal/vertical Gaussian filtering
with reflect padding. The blur has kernel 11 and sigma 7. Stored native dilation 2
is unused. Independent filtering is compared against the actual selected local
Kornia numerical definitions; unrelated ImageModule and shape-check wrappers
are narrowed for the tensor-only CPU oracle and disclosed in its receipt.

RIL updates detached centers using the entire batch before computing its loss.
Labels select intra margins; centers are not class-separated. The native inter
term and update/decay ordering are retained. Initial alpha 0.05 decays by 0.9 after
every batch_per_epoch RIL loss calls. Validation advances the model counter but
does not call RIL or update centers.

The local non-SWA train.py steps StepLR once after its complete epoch loop,
rather than every epoch. The trainer preserves that placement and saves a
source-scheduler.json receipt. The source iterates epochs 0..50 inclusively;
the remake example uses 51 sweeps. This does not establish original evaluation
cadence, sample order or full-protocol reproduction. Warmup and decay periods
remain explicit configuration values and are not inferred from loader length.

Complete state and strict import
--------------------------------

The native detector stores RIL in a plain dictionary, so its raw state_dict omits
RIL buffers and scalar counters. New task states preserve model tensors plus
forward count, feature centers, intra margins, alpha, RIL count and scheduler
state. Best selection restores these together; last and best snapshots retain
their respective method states. Shared trainer hooks preserve ordinary image
models' scoring and stateless behavior.

Raw initial_checkpoint import requires an explicit initial_method_state path.
The sidecar is a torch.save dictionary with these fields::

    {
        "forward_batches": source.batch_cnt,
        "ril": {
            "alpha": source.loss_func["ril"].alpha,
            "batch_cnt": source.loss_func["ril"].batch_cnt,
            "state_dict": source.loss_func["ril"].state_dict(),
        },
        "scheduler": None,  # or the actual captured StepLR state
    }

Phase and lost RIL state cannot be recovered from model tensors alone. Imports
do not invent them. A converted standalone checkpoint no longer needs either
source file. Actual ImageNet initialization instead uses explicit init_weights
with the existing native freshly replaced RGB stem; random construction requires
initialization: random. No missing-file fallback or automatic download occurs.

Validation scope
-----------------

The constructed native 256 resolution and batch 2 trajectory audits initialization and preprocessing,
warmup training, an explicitly positioned phase boundary, late inference and late
AGDA/RIL training. The RIL count is explicitly positioned at its decay boundary
to check alpha ordering without claiming 3591 prior updates. Outputs, augmentation,
gradients, weights, BN state, centers, counters and standalone predictions are
compared exactly. Raw states, inputs, RNG seed and source copies are preserved.
Actual ImageNet initialization is checked separately with seeded untrained heads.
The initial oracle attempt omitted a guard constant and failed in the blur path;
both that failure and the corrected successful run are retained.

The FFPP source-validation example retains RGB 256, mean/std 0.5, disabled data
augmentation, batch 32 and native optimizer/phase parameters. Remake manifests and
selection differ from the original full run. Historical trained scores, GPU
numerical alignment, exact optimizer/RNG resume and full-dataset reproduction
remain unverified. Receipts and reproducible commands are in
configs/migrations/validation.json. Use tools/bounded_run.py for model execution
and tools/test_cpu.py for complete CPU regression.
