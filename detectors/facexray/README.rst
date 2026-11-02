FaceXray model gate; training adapter pending
=============================================

FaceXrayDetector owns the native raw boundary decoder and classifier, composing
the independent lazy HRNetW48 backbone. Full native RGB256/model batch2 audits
agree byte-for-byte for constructor RNG, every initial state tensor, eval/train
features/logits/scores, CE plus1000 raw-mask MSE, parameter/image gradients, BN
state, complete Adam state/update and strict meta-restored scores. All1001
parameter tensors remain trainable, including66 unused backbone classification-
head tensors;935 receive gradients through features(). No legacy code is loaded
by production. The backbone imports Torch only and performs no IO.

Both interpolation conventions remain: HRNet concatenation uses native default
bilinear align_cornersFalse; the boundary decoder uses align_cornersTrue.
Raw boundary predictions may be negative and are not passed through sigmoid.
The whole mask256 is averaged to128x128 before FC1024/BN,128/BN and two logits.
The source-selected backbone is HRNetW48 despite the detector YAML B4 field.

This is a model-stage migration, not a complete detector integration. No Task
factory, Trainer or example experiment is exposed yet. The FFBlend nearest-face/
landmark bank, source/foreground selection, affine and triangular warp, color
transfer, boundary masks, synchronized shuffled pairs and per-view provenance
still need independent implementation and numerical verification. A generic
classification trainer would silently omit the active mask supervision and must
not be used. Constructed state is explicit; actual pretraining, historical
trained detector scores, exact optimizer/RNG resume, GPU alignment and full
dataset experiments remain unverified.

Native missing-checkpoint and missing-bank failures are preserved under
/var/lib/codex-experiments-naku/detector-rewrite-facexray-native-probe-20261102.
Full constructed audit states, gradients and optimizer are preserved under
/var/lib/codex-experiments-naku/detector-rewrite-facexray-model-20261102. Run
experiments.detector_rewrite.facexray_check reference then candidate on the same
fresh absolute case through tools/bounded_run.py, with distinct run directories.
The audit uses3GiB RAM, swap0, CPU2 and hidden GPU. These control-flow and small-
batch numerical checks do not establish complete training/data migration.

The official v1.0.2 bank assets have now been downloaded and inspected separately:
159199 finite float32(81,2) landmark arrays and31935 neighbor parents, each
with100 entries. All neighbor references resolve, but one self edge and69 arrays
outside0..255 are present. No silent filtering/clamping is authorized by this
inspection. Image availability, manifest bindings and generation still require
verification. Resource byte counts, hashes and failed downloads remain in
/var/lib/codex-experiments-naku/detector-rewrite-facexray-assets-20261102.

Native path-component6 accounting finds31935 original and127264 manipulated
background records in that landmark bank. Only original backgrounds have
neighbor entries. FFBlend trains over all landmark keys and uses self fallback
when a background has no neighbor entry; this differs from SBI real-only
selection. Future manifest transport must account for every foreground/background
exposure and source/validation separation explicitly.
