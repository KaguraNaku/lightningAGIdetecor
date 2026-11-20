"""Pinned GenD CLIP-L/14 task, with explicit head semantics on raw conversion."""

from importlib.resources import files
import json

import torch
from huggingface_hub import hf_hub_download

from detectors._shared.checkpoint import load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .checkpoint import read_gend_weights
from .detector import GenDDetector
from .encoder import GenDCLIPEncoder


MODEL_ID = "yermandy/GenD_CLIP_L_14"
REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"


def default_specification():
    return json.loads(
        files("configs").joinpath("migrations/gend_clip_l14.json").read_text()
    )


class GenDTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        fresh_backbone = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
        else:
            spec = default_specification()
            construction = {
                "vision_config": settings.get("vision_config", spec["vision_config"]),
                "head": settings.get("head", spec["head"]),
                "attention_implementation": settings.get(
                    "attention_implementation", "sdpa"
                ),
                "freeze_policy": "feature_extractor_norm_or_ln_names_plus_complete_head",
                "forward_precision": "float32_autocast_disabled",
                "training_profile": settings.get("training_profile", "author_ln_l2_ua"),
            }
            origin = {
                **reference_for("gend"),
                "objective": "cross_entropy",
                "backbone_family": "CLIP",
            }
            if settings.get("initial_checkpoint"):
                if "head" not in settings:
                    raise ValueError(
                        "Raw GenD imports require explicit head: linear or LinearNorm"
                    )
                weights, receipt = read_gend_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif settings.get("initialization") == "pretrained_backbone":
                from transformers import CLIPVisionModelWithProjection
                backbone_id = settings.get("backbone_id")
                if not backbone_id:
                    raise ValueError("Fresh GenD training requires backbone_id")
                fresh_backbone = CLIPVisionModelWithProjection.from_pretrained(
                    backbone_id, revision=settings.get("revision"), local_files_only=True,
                    attn_implementation=construction["attention_implementation"],
                )
                actual = fresh_backbone.config.to_dict()
                for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "image_size", "patch_size", "projection_dim"):
                    if actual[key] != construction["vision_config"][key]:
                        raise ValueError("GenD base CLIP geometry differs: " + key)
                origin.update(initialization="fresh_pretrained_backbone_random_head", backbone_id=str(backbone_id), revision=settings.get("revision"))
            elif settings.get("initialization") == "random":
                origin["initialization"] = "explicit_constructed_random"
            else:
                if (
                    settings.get("vision_config") is not None
                    or construction["head"] != spec["head"]
                ):
                    raise ValueError(
                        "Custom GenD geometry/head requires random initialization or explicit raw checkpoint"
                    )
                path = hf_hub_download(
                    MODEL_ID,
                    "model.safetensors",
                    revision=REVISION,
                    local_files_only=True,
                )
                weights, receipt = read_gend_weights(path)
                origin.update(
                    receipt,
                    pretrained_model={
                        "model_id": MODEL_ID,
                        "revision": REVISION,
                        "head": spec["head"],
                    },
                )
            preprocessing = preprocessing_config(
                construction["vision_config"]["image_size"],
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            if construction["training_profile"] == "author_ln_l2_ua":
                if construction["head"] != "LinearNorm":
                    raise ValueError("Full GenD requires its L2-normalized linear head")
                if settings.get("data_aug") is not None or settings.get("augmentation") is False:
                    raise ValueError("GenD author profile fixes its PIL augmentation policy")
                preprocessing.pop("data_aug", None)
                preprocessing["profile"] = "gend-author-pil-v1"
                preprocessing["augmentation"] = True
                origin.update(objective="CE+.5*uniformity+.1*label_alignment",
                              author_revision="a31acda1f8c6017b4753a8b39bfb9214127ad3b3",
                              protocol="author method under common train8/test32 source-only validation")
        if construction["attention_implementation"] not in ("eager", "sdpa"):
            raise ValueError("GenD attention implementation must be eager or sdpa")
        if (
            construction["freeze_policy"]
            != "feature_extractor_norm_or_ln_names_plus_complete_head"
            or construction["forward_precision"] != "float32_autocast_disabled"
        ):
            raise ValueError(
                "GenD state must retain native freezing and precision policy"
            )
        if construction.get("training_profile", "legacy_ce") not in ("legacy_ce", "author_ln_l2_ua"):
            raise ValueError("Unknown GenD training profile")
        if (construction.get("training_profile") == "author_ln_l2_ua"
                and (construction["head"] != "LinearNorm"
                     or preprocessing.get("profile") != "gend-author-pil-v1")):
            raise ValueError("Full GenD state requires its normalized head and author preprocessing")

        def build():
            encoder = GenDCLIPEncoder(
                vision_config=construction["vision_config"],
                attention_implementation=construction["attention_implementation"],
            )
            return GenDDetector(encoder, head=construction["head"])

        if weights is not None:
            with torch.device("meta"):
                model = build()
            load_strict(model, weights, assign=True)
            # Trainable tensors own their storage rather than aliasing an input
            # checkpoint or a caller's in-memory state dictionary.
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.detach().clone()
        else:
            model = build()
            if fresh_backbone is not None:
                model.model.feature_extractor.load_state_dict(fresh_backbone.state_dict(), strict=True)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def validate_training_parent(self, settings):
        expected = settings.get("training_profile", "author_ln_l2_ua")
        if self.construction.get("training_profile", "legacy_ce") != expected:
            raise ValueError("GenD parent training profile differs; a historical CE-only model cannot become full GenD silently")

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        if self.construction.get("training_profile", "legacy_ce") == "author_ln_l2_ua":
            from .trainer import GenDTrainer
            report = GenDTrainer(self.model, origin=self.origin).fit(
                loader, validation_loader=validation_loader, runtime=runtime,
                settings={"epochs": 30, **settings}, output_dir=output_dir)
            self._ready = True
            return report
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={"epochs": 11, **settings},
            output_dir=output_dir,
        )

    def get_collate_fn(self, stage):
        if self.preprocessing["profile"] == "gend-author-pil-v1":
            from data.loader import ImageCollator
            from .transforms import GenDTransform
            if stage not in ("train", "validation", "evaluation"):
                raise ValueError("Unknown GenD data stage")
            return ImageCollator(transform=GenDTransform(self.preprocessing, training=stage == "train"))
        return super().get_collate_fn(stage)
