FFD configured regression route
===============================

FFDTask composes the independent original Xception backbone, a regression-map
FFDDetector and FFDTrainer. Native RGB256 with mean/std0.5, no augmentation,
sigmoid separable728-to1 attention after block7, whole-batch BN, full backbone
training and CE are retained. The native map is16x16 at this input geometry.
Attention learns indirectly from classification. The unused adjust_channel
module remains trainable and serialized but receives no forward gradients.
Native defaults are Adam2e-4, weight decay5e-4, no scheduler and eleven sweeps.

The selected ffd.yaml sets maptype:reg and with_mask:false. Its optional mask
loss reads mask_pred while forward emits mask, so the unmodified masked branch
raises KeyError. This branch remains unconnected and is rejected explicitly;
the rewrite does not silently substitute a repaired mask loss. The dormant
none/template/PCA-template variants are not connected either. They require
separate specifications, native geometry checks and explicit repairs.

The original constructor reads ten external templates even for regression,
which never uses them. Production omits those unused reads without changing
model tensors or RNG order. Audits execute the native constructor, loading the
actual ten templates through a path adapter; selected original torchvision
numerical definitions supply conversion/normalization without importing the
unavailable current torchvision package. Only oracle telemetry is omitted.
Production never imports the legacy checkout or modifies sys.path.

Raw detector imports are strict. Task snapshots include the fixed recipe,
backbone/head/map/BN state, preprocessing and source/init identities. Meta
restoration requires no original files or pretrained assets. Historical trained
FFD scores, GPU alignment, exact optimizer/RNG resume and full dataset protocol
reproduction are separate unverified gates. The FFPP example differs from the
source's FF-NT selection. Optional mask supervision and alternate map variants
must not be described as migrated.

Evidence
--------

Constructed and actual ImageNet initialization audits at native RGB256/batch2
agree byte-for-byte for initialized model, preprocessing, attention maps,
features, logits, probabilities, CE, all parameter/image gradients, complete
BN/model state and named Adam state after one update. Restored score error is0.
All162 parameter tensors are trainable;158 receive gradients, with the four
unused adjust_channel parameter tensors retaining absent gradients. Complete
raw states, inputs, outputs, gradients, optimizer and hashed source snapshots
are preserved under persistent /var/lib/codex-experiments-naku, with home aliases.

Five FFD tests cover attention learning, unsupported mask/variant rejection,
strict raw import, standalone restoration and full native train/save/evaluate.
The workflow injects validation0.9/0.8 to exercise earlier-best selection; those
values are control-flow fixtures. All170 CPU tests pass in26 serial groups,
RAM3072MiB, swap0, CPU2 and GPUhidden. FFD-specific jobs peak below1.7GiB; the
full suite reaches3GiB with cache reclaim and no OOM. The initial missing-
torchvision oracle failure and native optional-mask failure are preserved.
configs/migrations/validation.json links raw artifacts, commands and receipts.
