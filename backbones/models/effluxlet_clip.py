"""CLIP vision skeleton/local initialization without discarded text-model state."""

from safetensors import safe_open
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel


def load_effluxlet_clip(
    model_id, *, vision_config, attention_implementation="sdpa", init_weights=None
):
    config = CLIPVisionConfig.from_dict(vision_config)
    config._attn_implementation = attention_implementation
    if init_weights is None:
        vision = CLIPVisionModel(config)
        return getattr(vision, "vision_model", vision)
    with torch.device("meta"):
        vision = CLIPVisionModel(config)
        vision = getattr(vision, "vision_model", vision)
    with safe_open(init_weights, framework="pt", device="cpu") as stream:
        weights = {
            name.removeprefix("vision_model."): stream.get_tensor(name)
            for name in stream.keys()
            if name.startswith("vision_model.")
        }
    positions = weights.pop("embeddings.position_ids", None)
    if positions is not None:
        expected = torch.arange(
            (config.image_size // config.patch_size) ** 2 + 1
        ).expand((1, -1))
        if not torch.equal(positions, expected):
            raise ValueError(
                "CLIP checkpoint has incompatible derived position indices"
            )
    vision.load_state_dict(weights, strict=True, assign=True)
    for module in vision.modules():
        if hasattr(module, "position_ids") and hasattr(module, "position_embedding"):
            module.position_ids = torch.arange(
                module.position_embedding.num_embeddings
            ).expand((1, -1))
    return vision
