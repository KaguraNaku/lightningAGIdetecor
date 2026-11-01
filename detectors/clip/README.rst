Independent CLIP baseline adapter
================================

CLIPTask composes the fully trainable vision encoder, a binary linear head and
the common whole-batch cross-entropy trainer. The local reference uses ViT-B/16
(768 hidden units, 12 layers, 224-pixel input), RGB cubic resize, mean/std 0.5 and
ordinary training augmentation. It does not freeze the backbone or insert LoRA.
The legacy YAML's Xception pretrained path is unused by its CLIP constructor.

Default construction requests the pinned B/16 pretrained vision weights through
the shared lazy loader, with local_files_only=True. Missing cached weights fail
explicitly. No random fallback or automatic download is introduced. To request
from-scratch initialization, set initialization=random. A vision_config override
is accepted for explicit random construction or raw checkpoint import; it is
saved in the resulting task and can describe a separate experimental variant.

The packaged configs/migrations/clip_b16_vision.json records the exact upstream
configuration URL, revision and hash. This small descriptor was retrieved from
the official model; pretrained weight tensors were not downloaded in this
milestone. Its pinned source is:

https://huggingface.co/openai/clip-vit-base-patch16/raw/57c216476eefef5ab752ec549e440a49ae4ae5f3/config.json

Raw full-detector weights import strictly. The common checkpoint converter can
produce a standalone engine checkpoint; native B/16 imports use the packaged
geometry without fetching metadata. Non-native raw weights need an explicit
vision_config. Saved tasks restore their geometry, attention implementation,
preprocessing and weights without access to any original weight/cache files.

The example clip_ffpp.yaml retains native batch 32, Adam defaults and 10 epochs,
but uses the remake manifest and source-validation selection policy. It is not
an original-protocol reproduction config. All execution should go through
tools/bounded_run.py; do not silently split a batch to fit a memory limit.

Validation gates are distinct: CPU tests use a tiny explicit geometry for the
full train/save/restore/evaluate workflow. A separate audit uses full B/16
geometry, seeded constructed random vision weights and batch 1, comparing the
actual local detector definitions with the rewrite. Cached pretrained B/16
initialization, historical trained-detector scores and full-dataset reproduction
remain unverified. The first reference attempt failed because the installed
Transformers exposes CLIPVisionModel directly rather than under vision_model;
the corrected oracle handles both layouts. The failed receipt is preserved.
