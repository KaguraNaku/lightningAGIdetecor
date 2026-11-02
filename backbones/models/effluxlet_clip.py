"""Compatibility entry for the shared local CLIP vision checkpoint loader."""

from .clip_checkpoint import load_clip_checkpoint


def load_effluxlet_clip(model_id, **kwargs):
    return load_clip_checkpoint(model_id, **kwargs)
