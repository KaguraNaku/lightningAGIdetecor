"""Full CLIP fine-tuning through the common image-classification task."""

from importlib.resources import files
import json

import torch
from transformers import AutoConfig, CLIPVisionConfig, CLIPVisionModel

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

from .detector import CLIPDetector


def default_specification():
    return json.loads(
        files("configs").joinpath("migrations/clip_b16_vision.json").read_text()
    )


class CLIPTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
            cfg = CLIPVisionConfig.from_dict(construction["vision_config"])
            attention = construction["attention_implementation"]
        else:
            spec = default_specification()
            model_id = settings.get("backbone_id", spec["model_id"])
            attention = settings.get("attention_implementation", "sdpa")
            if attention not in ("sdpa", "eager"):
                raise ValueError("CLIP attention implementation must be sdpa or eager")
            options = {"local_files_only": settings.get("local_files_only", True)}
            revision = settings.get(
                "revision", spec["revision"] if model_id == spec["model_id"] else None
            )
            if revision is not None:
                options["revision"] = revision
            origin = {**reference_for("clip"), "objective": "cross_entropy"}
            checkpoint = settings.get("initial_checkpoint")
            random = settings.get("initialization") == "random"
            if checkpoint or random:
                if settings.get("vision_config") is not None:
                    cfg = CLIPVisionConfig.from_dict(settings["vision_config"])
                elif model_id == spec["model_id"]:
                    cfg = CLIPVisionConfig.from_dict(spec["vision_config"])
                else:
                    cfg = AutoConfig.from_pretrained(model_id, **options)
                    cfg = getattr(cfg, "vision_config", cfg)
                if checkpoint:
                    weights, receipt = read_weights(checkpoint)
                    origin.update(receipt)
                else:
                    cfg._attn_implementation = attention
                    vision = CLIPVisionModel(cfg)
                    origin["initialization"] = "explicit_random"
            else:
                if settings.get("vision_config") is not None:
                    raise ValueError(
                        "vision_config requires initialization: random or an initial_checkpoint"
                    )
                vision = load_backbone(
                    model_id,
                    "vision",
                    model_type="clip",
                    attn_implementation=attention,
                    **options,
                )
                cfg = vision.config
                origin["pretrained_vision"] = {
                    "model_id": model_id,
                    "revision": revision,
                }
            construction = {
                "vision_config": cfg.to_dict(),
                "attention_implementation": attention,
            }
            preprocessing = preprocessing_config(
                cfg.image_size,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
        if weights is not None:
            cfg._attn_implementation = attention
            with torch.device("meta"):
                model = CLIPDetector(CLIPVisionModel(cfg))
            load_strict(model, weights, assign=True)
        else:
            model = CLIPDetector(vision)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )


def create_task(settings, *, state=None):
    return CLIPTask(settings, state=state)
