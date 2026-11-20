"""Loader for SigLIP and fixed-resolution SigLIP2's SigLIP architecture."""


def load_siglip_vision(model_id, **kwargs):
    from transformers import SiglipVisionModel

    return SiglipVisionModel.from_pretrained(model_id, **kwargs)
