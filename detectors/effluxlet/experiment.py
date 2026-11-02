"""Effluxlet Task retains native model math and an explicit optimizer repair."""

from importlib.resources import files
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch

from backbones.load_backbone import load_backbone
from data.manifest import sha256
from detectors._shared.checkpoint import read_weights, load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .detector import EffluxletDetector
from .trainer import EffluxletTrainer


RECIPE = {
    "rank": 32,
    "scale": 1.0,
    "wavelet": "haar",
    "wavelet_mode": "zero",
    "levels": 4,
    "cdc_dilations": [1, 2, 3, 4],
    "cdc_channels": 9,
    "directions": 3,
    "fusion": "sum_global_vectors",
    "orthogonal_loss": "mean_absolute_cosine_main_and_BA_times100",
    "optimizer_bridge": "disjoint_all_except_wavelets_and_wavelets",
}


def default_specification():
    return json.loads(
        files("configs").joinpath("migrations/effluxlet_clip_l14.json").read_text()
    )


class EffluxletTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        initial = None
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
                "recipe": RECIPE,
                "vision_config": settings.get("vision_config", spec["vision_config"]),
                "attention_implementation": settings.get(
                    "attention_implementation", "sdpa"
                ),
            }
            origin = {
                **reference_for("effluxlet"),
                "objective": RECIPE["orthogonal_loss"],
                "optimizer_bridge": "Source generic optimizer fails on nested config; explicit all-except-wavelets/wavelets groups retain configured rates",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif settings.get("initialization") == "random":
                origin["initialization"] = "explicit_constructed_random"
            else:
                if settings.get("vision_config") is not None:
                    raise ValueError(
                        "Custom Effluxlet geometry requires random initialization or a full initial_checkpoint"
                    )
                initial = settings.get("init_weights") or hf_hub_download(
                    spec["model_id"],
                    "model.safetensors",
                    revision=spec["revision"],
                    local_files_only=True,
                )
                initial = Path(initial).expanduser().resolve()
                origin["pretrained_vision"] = {
                    "path": str(initial),
                    "sha256": sha256(initial),
                    "model_id": spec["model_id"],
                    "revision": spec["revision"],
                }
            preprocessing = preprocessing_config(
                construction["vision_config"]["image_size"],
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
        if construction["recipe"] != RECIPE:
            raise ValueError(
                "Effluxlet state must retain native model and explicit bridge recipe"
            )
        if construction["attention_implementation"] not in ("sdpa", "eager"):
            raise ValueError("Effluxlet attention must be sdpa or eager")

        def build():
            vision = load_backbone(
                "effluxlet_clip",
                "vision",
                model_type="effluxlet_clip",
                vision_config=construction["vision_config"],
                attention_implementation=construction["attention_implementation"],
                init_weights=initial,
            )
            return EffluxletDetector(vision)

        if weights is not None:
            with torch.device("meta"):
                model = build()
            load_strict(model, weights, assign=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.detach().clone()
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
        report = EffluxletTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={"epochs": 16, **settings},
            output_dir=output_dir,
        )
        self._ready = True
        return report


def create_task(settings, *, state=None):
    return EffluxletTask(settings, state=state)
