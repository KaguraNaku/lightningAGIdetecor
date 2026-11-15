"""Unified Task composes SLADD Detector, explicit data formats and own Trainer."""

from copy import deepcopy
from pathlib import Path

import torch

from data.loader import ImageCollator
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import ImageTransform, preprocessing_config
from .checkpoint import (
    detector_weights,
    require_recipe,
    restore_detector,
    selected_recipe,
)
from .dataset import SLADDPairCollator, SLADDPairDataset
from .images import SLADDImageDataset
from .initialization import create_detector
from .protocol import PROFILES
from .trainer import SLADDTrainer
from .transforms import AUGMENTATION


def same_metadata(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            same_metadata(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_metadata(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def preprocessing_for(profile):
    if profile not in PROFILES:
        raise ValueError(
            "SLADD requires explicit source_filesystem_gray or source_lmdb_rgb_bytes data_profile"
        )
    return {
        "data_profile": profile,
        "generic": preprocessing_config(256, [0.5] * 3, [0.5] * 3, augmentation=False),
        "train_augmentation": deepcopy(AUGMENTATION),
        "train_data": "Fake indexed,real sampled with replacement;native fake-first decode/augmentation andreal-all/fake-all collation;policy switching andgenerated flipviews inDetector",
        "with_mask": True,
        "with_landmark": True,
        "evaluation_augmented": False,
        "validation_policy": "Source val only;native step cadence,not originalmulti-dataset selection or8-worker RNG",
        "gray_profile_failure_retained": profile == "source_filesystem_gray",
        "lmdb_database_IO_reproduced": False,
        "byte_profile_is_lmdb_True_decode_override": profile == "source_lmdb_rgb_bytes",
    }


class SLADDTask(ImageClassificationTask):
    prediction_unit = "image"

    def __init__(self, settings, *, state=None):
        recipe = selected_recipe()
        require_recipe(recipe)
        if set(settings) - (
            set(recipe)
            | {"data_profile", "initial_checkpoint", "init_weights", "score_type"}
        ):
            raise ValueError("Unknown SLADD model setting")
        if settings.get("score_type", "probability") != "probability":
            raise ValueError("SLADD Task emits fake-class probabilities")
        for key, value in recipe.items():
            if key in settings and not same_metadata(settings[key], value):
                raise ValueError(
                    f"Requested SLADD construction differs from native recipe:{key}"
                )
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin = (
                deepcopy(state[name])
                for name in ("construction", "preprocessing", "origin")
            )
            self.validate_metadata(construction, preprocessing, origin)
            if "data_profile" in settings and not same_metadata(
                settings["data_profile"], construction["data_profile"]
            ):
                raise ValueError("Requested SLADD data profile differs from saved Task")
            model = restore_detector(state["state_dict"], recipe=construction["recipe"])
            ready = True
        else:
            profile = settings.get("data_profile")
            preprocessing = preprocessing_for(profile)
            construction = {
                "recipe": recipe,
                "prediction_unit": "image",
                "data_profile": profile,
            }
            origin = {
                **reference_for("sladd"),
                "source_only_selection": True,
                "exact_optimizer_rng_resume_supported": False,
                "full_experiment_reproduction": False,
                "preprocessing_note": "Formal filesystem gray failure retained;byte format isexplicit source-lmdb override,not anauthor recipe oractual LMDB IO",
            }
            initial, pretrained = (
                settings.get("initial_checkpoint"),
                settings.get("init_weights"),
            )
            if (initial is None) == (pretrained is None):
                raise ValueError(
                    "Select one complete SLADD initial_checkpoint or native ImageNet init_weights; no implicit random source initialization"
                )
            if initial is not None:
                weights, imported = detector_weights(initial, recipe=recipe)
                model = restore_detector(weights, recipe=recipe)
                origin.update(imported)
                ready = True
            else:
                model = create_detector(pretrained)
                origin["native_initialization"] = deepcopy(model.initialization)
                ready = False
        self.validate_metadata(construction, preprocessing, origin)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=ready,
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
            raise ValueError("Incomplete or unsupported SLADD Task checkpoint")

    @staticmethod
    def validate_metadata(construction, preprocessing, origin):
        if (
            not isinstance(construction, dict)
            or set(construction) != {"recipe", "prediction_unit", "data_profile"}
            or construction["prediction_unit"] != "image"
        ):
            raise ValueError(
                "SLADD Task requires native recipe and explicit image/data profile"
            )
        require_recipe(construction["recipe"])
        if not same_metadata(
            preprocessing, preprocessing_for(construction["data_profile"])
        ):
            raise ValueError(
                "SLADD preprocessing differs from saved explicit native profile"
            )
        if (
            not isinstance(origin, dict)
            or origin.get("method") != "sladd"
            or origin.get("source_only_selection") is not True
            or origin.get("exact_optimizer_rng_resume_supported") is not False
            or origin.get("full_experiment_reproduction") is not False
        ):
            raise ValueError(
                "SLADD Task requires explicit source-selection and non-resume provenance"
            )

    def prepare_dataset(self, dataset, *, stage):
        if stage == "train":
            return SLADDPairDataset(dataset, profile=self.construction["data_profile"])
        return SLADDImageDataset(dataset, stage=stage)

    def get_collate_fn(self, stage):
        if stage == "train":
            return SLADDPairCollator()
        if stage not in {"validation", "evaluation"}:
            raise ValueError("Unknown SLADD preprocessing stage")
        return ImageCollator(
            transform=ImageTransform(self.preprocessing["generic"], training=False)
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        def started():
            self._ready = False

        result = SLADDTrainer(
            self.model,
            data_profile=self.construction["data_profile"],
            origin=self.origin,
            on_start=started,
        ).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        protocol = Path(output_dir) / result["protocol"]
        self.origin["training_protocol"] = {
            "path": str(protocol),
            "requested_settings": deepcopy(settings),
            "selected": deepcopy(result["selected"]),
        }
        return result

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import complete SLADD weights before prediction")
        if runtime.precision != "fp32" or images.dtype != torch.float32:
            raise ValueError(
                "Selected SLADD prediction profile requires fp32 images/runtime"
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
            raise ValueError("SLADD requires one finite fake probability per image")
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


def create_task(settings, *, state=None):
    return SLADDTask(settings, state=state)
