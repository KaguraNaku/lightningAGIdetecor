"""ProDet Task composes native model, explicit data recipe and dedicated Trainer."""

from copy import deepcopy
from pathlib import Path
import re

import torch

from backbones.load_backbone import load_backbone
from data.loader import ImageCollator
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import ImageTransform, preprocessing_config
from detectors._shared.initialization import EfficientNetB4Initializer
from .checkpoint import (
    detector_weights,
    require_recipe,
    restore_detector,
    selected_recipe,
)
from .dataset import ProDetDataset, ProDetDataCollator
from .detector import ProDetDetector
from .images import ProDetImageDataset
from .trainer import ProDetTrainer


PROFILES = {"native_entrypoint", "specialized_quadruples"}


def preprocessing_for(profile):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError(
            "Select ProDet native_entrypoint or specialized_quadruples explicitly"
        )
    return {
        "data_profile": profile,
        "generic": preprocessing_config(
            256,
            (0.5,) * 3,
            (0.5,) * 3,
            data_aug={"rotate_prob": 0.0, "quality_lower": 40, "quality_upper": 100},
        ),
        "train": "native_generic_binary24"
        if profile == "native_entrypoint"
        else "explicit_SBIPlusV2_whole_face_softBI24quadruples_to96",
        "validation_augmented": False,
        "evaluation_augmented": False,
        "with_landmark": profile == "specialized_quadruples",
        "with_mask": False,
        "validation_role": "remake_source_val_with_native_test_mode_pixels;not_target_selection",
        "worker_RNG": "remake_seeded_workers;not_original_upstream_or8worker_global_RNG",
    }


def same_metadata(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            same_metadata(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_metadata(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_neighbors(value):
    if value is None:
        return
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256", "bytes"}
        or not isinstance(value["path"], str)
        or not Path(value["path"]).is_absolute()
        or not isinstance(value["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        or type(value["bytes"]) is not int
        or value["bytes"] <= 0
    ):
        raise ValueError(
            "ProDet training_neighbors requires absolute path/hash/size identity"
        )


class ProDetTask(ImageClassificationTask):
    prediction_unit = "image"

    def __init__(self, settings, *, state=None):
        recipe = selected_recipe()
        require_recipe(recipe)
        allowed = set(recipe) | {
            "data_profile",
            "training_neighbors",
            "initial_checkpoint",
            "init_weights",
            "initialization",
            "score_type",
        }
        if (
            set(settings) - allowed
            or settings.get("score_type", "probability") != "probability"
        ):
            raise ValueError("Unknown ProDet settings or non-native score type")
        for key, value in recipe.items():
            if key in settings and not same_metadata(settings[key], value):
                raise ValueError(
                    f"ProDet requested non-native construction setting:{key}"
                )
        weights = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin = (
                deepcopy(state[key])
                for key in ("construction", "preprocessing", "origin")
            )
            weights = state["state_dict"]
            self.validate_metadata(construction, preprocessing, origin)
            for key in ("data_profile", "training_neighbors"):
                if key in settings and not same_metadata(
                    settings[key], construction[key]
                ):
                    raise ValueError(
                        "ProDet requested data bindings differ from saved Task"
                    )
        else:
            profile = settings.get("data_profile")
            preprocessing = preprocessing_for(profile)
            construction = {
                "recipe": recipe,
                "prediction_unit": "image",
                "data_profile": profile,
                "training_neighbors": deepcopy(settings.get("training_neighbors")),
            }
            origin = {
                **reference_for("prodet"),
                "source_only_selection": True,
                "exact_optimizer_rng_resume_supported": False,
                "full_experiment_reproduction": False,
                "data_recipe_note": "native_entrypoint preserves actual generic selector;specialized_quadruples is explicit SBIPlusV2 wiring repair",
            }
            self.validate_metadata(construction, preprocessing, origin)
            initial = settings.get("initial_checkpoint")
            if initial is not None:
                if (
                    settings.get("init_weights") is not None
                    or settings.get("initialization") is not None
                ):
                    raise ValueError("Select one ProDet initialization source")
                weights, receipt = detector_weights(initial, recipe=recipe)
                origin.update(receipt)
            elif (
                settings.get("init_weights") is None
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "ProDet requires full initial_checkpoint,ImageNet init_weights,or explicit initialization=random"
                )
            elif (
                settings.get("init_weights") is not None
                and settings.get("initialization") is not None
            ):
                raise ValueError("Select one ProDet initialization source")
        if weights is not None:
            model = restore_detector(weights, recipe=construction["recipe"])
        else:
            backbone = load_backbone(
                "efficientnetb4",
                "vision",
                model_type="efficientnetb4",
                num_classes=2,
                in_channels=3,
                mode="original",
                dropout=0.0,
            )
            if settings.get("init_weights") is not None:
                origin["pretrained_backbone"] = EfficientNetB4Initializer.initialize(
                    backbone, settings["init_weights"]
                )
            else:
                origin["initialization"] = "explicit_random;not_training"
            model = ProDetDetector(backbone)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    @staticmethod
    def validate_state(state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {"version", "construction", "preprocessing", "origin", "state_dict"}
            or type(state["version"]) is not int
            or state["version"] != 1
        ):
            raise ValueError("Unsupported or incomplete ProDet Task state")

    @staticmethod
    def validate_metadata(construction, preprocessing, origin):
        if (
            not isinstance(construction, dict)
            or set(construction)
            != {"recipe", "prediction_unit", "data_profile", "training_neighbors"}
            or construction["prediction_unit"] != "image"
            or not isinstance(origin, dict)
        ):
            raise ValueError(
                "ProDet Task requires complete native construction/data metadata"
            )
        require_recipe(construction["recipe"])
        expected = preprocessing_for(construction["data_profile"])
        if not same_metadata(preprocessing, expected):
            raise ValueError(
                "ProDet saved preprocessing differs from its explicit native profile"
            )
        validate_neighbors(construction["training_neighbors"])
        if (
            construction["data_profile"] == "native_entrypoint"
            and construction["training_neighbors"] is not None
        ):
            raise ValueError(
                "ProDet generic entrypoint does not use a neighbor sidecar"
            )

    def prepare_dataset(self, dataset, *, stage):
        if (
            stage == "train"
            and self.construction["data_profile"] == "specialized_quadruples"
        ):
            if self.construction["training_neighbors"] is None:
                raise ValueError(
                    "ProDet specialized training requires explicit training_neighbors"
                )
            return ProDetDataset(
                dataset, neighbors=self.construction["training_neighbors"]
            )
        return ProDetImageDataset(dataset, stage=stage)

    def get_collate_fn(self, stage):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown ProDet preprocessing stage")
        if (
            stage == "train"
            and self.construction["data_profile"] == "specialized_quadruples"
        ):
            return ProDetDataCollator()
        return ImageCollator(
            transform=ImageTransform(self.preprocessing["generic"], stage == "train")
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._ready = False
        result = ProDetTrainer(
            self.model,
            data_profile=self.construction["data_profile"],
            origin=self.origin,
        ).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return result

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import complete ProDet weights before prediction")
        if runtime.precision != "fp32" or images.dtype != torch.float32:
            raise ValueError(
                "The selected ProDet prediction alignment profile requires fp32"
            )
        self.model.to(runtime.device).eval()
        scores = self.model.forward_outputs(
            images.to(runtime.device, non_blocking=True), inference=True
        )["prob"]
        if (
            scores.shape != (len(images),)
            or not torch.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
        ):
            raise ValueError("ProDet requires one finite fake probability per image")
        return scores

    def state_dict(self):
        self.validate_metadata(self.construction, self.preprocessing, self.origin)
        state = super().state_dict()
        return {
            **state,
            "construction": deepcopy(self.construction),
            "preprocessing": deepcopy(self.preprocessing),
            "origin": deepcopy(self.origin),
        }
