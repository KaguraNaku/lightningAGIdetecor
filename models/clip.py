"""Selected CLIP vision component; importing this module loads no model library."""


def load_clip_vision(model_id, **kwargs):
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(model_id, **kwargs)


def initialize_clip_position_ids(model):
    """Materialize CLIP's nonpersistent position indices after a meta load."""
    import torch

    for module in model.modules():
        if hasattr(module, "position_ids") and hasattr(module, "position_embedding"):
            module.position_ids = torch.arange(
                module.position_embedding.num_embeddings, device="cpu",
            ).unsqueeze(0)
