"""LoRA adapter for the shared experiment entry point."""

import torch
from transformers import AutoConfig, CLIPVisionConfig, CLIPVisionModel

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors._shared.provenance import reference_for

from .checkpoint import import_legacy
from .detector import LoRADetector

MODEL_ID = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


class LoRATask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        origin = {**reference_for("lora"), "objective": "cross_entropy"}
        weights = None
        if state is not None:
            self.validate_state(state)
            construction = state["construction"]
            if construction.get("weight_mode") != "unmerged":
                raise ValueError("Unsupported LoRA task weight mode")
            cfg = CLIPVisionConfig.from_dict(construction["vision_config"])
            attention = construction["attention_implementation"]
            rank, alpha = construction["rank"], construction["alpha"]
            preprocessing, origin, weights = (
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
        else:
            model_id = settings.get("backbone_id", MODEL_ID)
            rank, alpha = settings.get("lora_r", 16), settings.get("lora_alpha", 16)
            attention = settings.get("attention_implementation", "sdpa")
            if attention not in ("sdpa", "eager"):
                raise ValueError("LoRA attention implementation must be sdpa or eager")
            options = {"local_files_only": settings.get("local_files_only", False)}
            revision = settings.get(
                "revision", REVISION if model_id == MODEL_ID else None
            )
            if revision is not None:
                options["revision"] = revision
            if settings.get("initial_checkpoint"):
                cfg = AutoConfig.from_pretrained(model_id, **options)
                cfg = getattr(cfg, "vision_config", cfg)
                weights, receipt = import_legacy(
                    settings["initial_checkpoint"],
                    alpha=alpha,
                    rank=rank,
                    weight_mode=settings.get("legacy_weight_mode"),
                )
                origin.update(receipt)
            else:
                vision = load_backbone(
                    model_id, "vision", attn_implementation=attention, **options
                )
                cfg = vision.config
            preprocessing = preprocessing_config(
                cfg.image_size,
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
            construction = {
                "vision_config": cfg.to_dict(),
                "rank": rank,
                "alpha": alpha,
                "attention_implementation": attention,
                "weight_mode": "unmerged",
            }
        if weights is not None:
            cfg._attn_implementation = attention
            with torch.device("meta"):
                model = LoRADetector(CLIPVisionModel(cfg), rank=rank, alpha=alpha)
            load_strict(model, weights, assign=True)
        else:
            model = LoRADetector(vision, rank=rank, alpha=alpha)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )


def create_task(settings, *, state=None):
    return LoRATask(settings, state=state)
