"""ProDet Task composes native model, explicit data recipe and dedicated Trainer."""

from copy import deepcopy
from pathlib import Path
import re

import torch

from data.loader import ImageCollator
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import ImageTransform, preprocessing_config
from .backbone import EfficientNetB4
from .checkpoint import (
    detector_weights,
    require_recipe,
    restore_detector,
    selected_recipe,
)
from .dataset import ProDetDataset, ProDetDataCollator
from .detector import ProDetDetector
from .images import ProDetImageDataset
from .initialization import EfficientNetB4Initializer
from .trainer import ProDetTrainer
from .protocol import neighbor_identity


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
        self._validate_model_settings(settings, recipe)
        if state is None:
            construction, preprocessing, origin = self._training_config(settings, recipe)
            model, initialized = self._initialize_model(settings, recipe, origin)
        else:
            construction, preprocessing, origin, weights = self._restore_config(settings, state)
            model = restore_detector(weights, recipe=construction["recipe"])
            initialized = True

        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=initialized,
        )

    @staticmethod
    def _validate_model_settings(settings, recipe):
        options = {
            "data_profile", "training_neighbors", "initial_checkpoint",
            "init_weights", "initialization", "score_type",
        }
        if extra := set(settings) - recipe.keys() - options:
            raise ValueError(f"Unsupported ProDet settings: {', '.join(sorted(extra))}")
        if settings.get("score_type", "probability") != "probability":
            raise ValueError("ProDet outputs a fake-class probability")
        for key, value in recipe.items():
            if key in settings and not same_metadata(settings[key], value):
                raise ValueError(f"ProDet requested non-native construction setting: {key}")

    def _training_config(self, settings, recipe):
        profile = settings.get("data_profile")
        preprocessing = preprocessing_for(profile)
        neighbors = settings.get("training_neighbors")
        if isinstance(neighbors, str):
            neighbors = neighbor_identity(neighbors)
        construction = {
            "recipe": recipe,
            "prediction_unit": "image",
            "data_profile": profile,
            "training_neighbors": deepcopy(neighbors),
        }
        origin = {
            **reference_for("prodet"),
            "source_only_selection": True,
            "exact_optimizer_rng_resume_supported": False,
            "full_experiment_reproduction": False,
            "data_recipe_note": "specialized_quadruples executes the author SBIPlusV2 four-domain recipe;native_entrypoint is historical binary compatibility",
        }
        if profile == "specialized_quadruples":
            origin["author_method"] = {
                "repository": "https://github.com/beautyremain/ProDet",
                "revision": "050856249569f8ae6de4e847265f9d6e2139b3b2",
                "data": "SBIPlusV2 real/blendfake/bifake/deepfake;24quadruples->96views",
                "controls": "strict train8/test32;source-only validation;train-only nearest pool",
                "identical_paper_run_claimed": False,
            }
        self.validate_metadata(construction, preprocessing, origin)
        return construction, preprocessing, origin

    def _restore_config(self, settings, state):
        construction, preprocessing, origin, weights = self.unpack_state(state)
        self.validate_metadata(construction, preprocessing, origin)
        if "data_profile" in settings and not same_metadata(
            settings["data_profile"], construction["data_profile"]
        ):
            raise ValueError("ProDet requested data bindings differ from saved Task")
        if "training_neighbors" in settings:
            requested = settings["training_neighbors"]
            saved = construction["training_neighbors"]
            if isinstance(requested, str):
                # Restoring a model only needs the saved sidecar path.
                path = str(Path(requested).expanduser().resolve())
                if not isinstance(saved, dict) or path != saved["path"]:
                    raise ValueError("ProDet requested neighbor path differs from saved Task")
            elif not same_metadata(requested, saved):
                raise ValueError("ProDet requested data bindings differ from saved Task")
        return deepcopy(construction), deepcopy(preprocessing), deepcopy(origin), weights

    @staticmethod
    def _initialize_model(settings, recipe, origin):
        checkpoint = settings.get("initial_checkpoint")
        imagenet = settings.get("init_weights")
        initialization = settings.get("initialization")
        if sum(value is not None for value in (checkpoint, imagenet, initialization)) > 1:
            raise ValueError("Select one ProDet initialization source")
        if checkpoint is not None:
            weights, receipt = detector_weights(checkpoint, recipe=recipe)
            origin.update(receipt)
            return restore_detector(weights, recipe=recipe), True
        if imagenet is None and initialization != "random":
            raise ValueError(
                "ProDet requires initial_checkpoint, ImageNet init_weights, or initialization: random"
            )

        backbone = EfficientNetB4(mode="original")
        if imagenet is not None:
            origin["pretrained_backbone"] = EfficientNetB4Initializer.initialize(backbone, imagenet)
        else:
            origin["initialization"] = "explicit_random;not_training"
        return ProDetDetector(backbone), False

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
            if dataset.header["dataset"] == "FaceForensics++":
                from engine.protocol import VideoSamplingPolicy
                VideoSamplingPolicy({"name": "train8_test32", "train_frames": 8,
                                     "test_frames": 32}).validate(dataset.header, dataset.records)
            if self.construction["training_neighbors"] is None:
                raise ValueError(
                    "ProDet specialized training requires explicit training_neighbors"
                )
            return ProDetDataset(
                dataset, neighbors=self.construction["training_neighbors"]
            )
        return ProDetImageDataset(dataset, stage=stage)

    def build_collator(self, *, training):
        if training and self.construction["data_profile"] == "specialized_quadruples":
            return ProDetDataCollator()
        return ImageCollator(transform=ImageTransform(self.preprocessing["generic"], training))

    def build_trainer(self):
        return ProDetTrainer(
            self.model,
            data_profile=self.construction["data_profile"],
            origin=self.origin,
        )

    def fit(self, loader, **kwargs):
        self._ready = False
        return super().fit(loader, **kwargs)

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

    def checkpoint_metadata(self):
        self.validate_metadata(self.construction, self.preprocessing, self.origin)
        return {
            key: deepcopy(value) for key, value in super().checkpoint_metadata().items()
        }
