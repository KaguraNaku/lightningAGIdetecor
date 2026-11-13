# Forensics Adapter integration

Source: https://github.com/OUC-VAS/ForensicsAdapter
Revision: a4cc94fa4255d2baa85cb26989fb884b098010e3.
Source file digests are in reference.json. Adapted author code is CC BY-NC4.0;
the small timm0.6.12 operator subset retains its Apache2.0 license separately.

The detector composes a frozen author CLIP backbone, TinyViT query adapter,
reconstructed attention, boundary decoder and classifier. Task, dataset,
collator, loss and trainer remain separate objects. No target label or mask
enters classification inference. Full native source training requires explicit
mask supervision; missing fake masks are fatal rather than zero-filled.

Fake training rows need manifest metadata:
```json
{"forada_mask": {"path": "relative/mask.png", "sha256": "actual_sha256"}}
```
Paths are confined to the shared dataset root and checked before binding and on
use. Real source masks are zero. The original float32 OpenCV patch/boundary
operations are preserved, including its bitwise-not behavior. Augmentation=False
is the official executable default; the broken augmented-target route is not
silently replaced by another recipe.

The objective is 10*CE + 200*boundary-MSE + 20*adapter-intra + 10*CLIP-contrastive.
Native all-real batches produce undefined contrastive loss; the adapter fails
clearly instead of reporting a finite CE-only substitute. Whole batches retain
BatchNorm and contrastive semantics.

## Explicit reference repairs and retained behavior

1. The original reconstructed-attention patch slice write overwrites a tensor
   saved for backward on CPU FP32. The unmodified failure is preserved in the
   run artifacts. An out-of-place concatenation implements the same residual
   expression and permits backward.
2. The original classifier's unqualified squeeze drops N=1. Squeezing the channel
   dimension instead preserves two logits per image, without changing N>=2 math.
3. The original adapter creates unregistered random Fusion layers on every
   forward, including evaluation. This unusual behavior is retained rather than
   changed into a new learned persistent fusion scheme. Exact prediction replay
   requires matching external Torch RNG state as well as weights. Training
   snapshots record RNG; exact restart/resume is not implemented by the shared
   trainer. Parameter checkpoint round trips are strict.

TinyViT uses a small dependency-free subset of timm0.6.12 operators because that
timm release cannot import on the project's Python3.13. Loaded state keys and
forward/loss/gradient/Adam-step equality are checked against actual timm0.6.12
in a separate Python3.10 reference environment.

Differential fixtures retain 24 CLIP layers, 128 queries and native spatial
geometry256/224, with CLIP width reduced to64 and one attention head for bounded
checks. This is not full-width CLIP-L/14 training or trained-score reproduction.
The production configuration remains CLIP-L/14 (1024width,16heads).
Only FP32 is validated; no mixed-precision parity claim is made.

Initialization is explicit: base_clip_checkpoint for fresh source fitting,
initial_checkpoint for a complete author detector state, or initialization=random
for constructed fixtures. There is no implicit network download or random
fallback when an asset is absent.
