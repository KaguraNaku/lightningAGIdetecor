"""Pinned GenD CLIP-L/14 task, with explicit head semantics on raw conversion."""

from importlib.resources import files
import json

import torch
from huggingface_hub import hf_hub_download

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .checkpoint import read_gend_weights
from .detector import GenDDetector


MODEL_ID = "yermandy/GenD_CLIP_L_14"
REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"


def default_specification():
    return json.loads(
        files("configs").joinpath("migrations/gend_clip_l14.json").read_text()
    )


class GenDTask(ImageClassificationTask):
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

        def build():
            encoder = load_backbone(
                "gend_clip",
                "vision",
                model_type="gend_clip",
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
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={"epochs": 11, **settings},
            output_dir=output_dir,
        )


def create_task(settings, *, state=None):
    return GenDTask(settings, state=state)
