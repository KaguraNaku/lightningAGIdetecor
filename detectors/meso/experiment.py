"""Meso variants composed with the shared source-validation task and trainer."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.detector import BackboneClassifier
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

VARIANTS = ("meso4", "meso4Inception")


class MesoTask(ImageClassificationTask):
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
            variant = settings.get("variant", "meso4")
            if variant not in VARIANTS:
                raise ValueError(f"Meso variant must be one of {VARIANTS}")
            construction = {"variant": variant, "num_classes": 2, "in_channels": 3}
            # Both local configs explicitly disable train augmentation.
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {**reference_for(variant), "objective": "cross_entropy"}
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            else:
                origin["initialization"] = "native_random"
        variant = construction["variant"]
        if variant not in VARIANTS:
            raise ValueError("Unsupported Meso checkpoint variant")
        options = {k: v for k, v in construction.items() if k != "variant"}
        if weights is not None:
            with torch.device("meta"):
                model = BackboneClassifier(
                    load_backbone(variant, "vision", model_type=variant, **options)
                )
            load_strict(model, weights, assign=True)
        else:
            model = BackboneClassifier(
                load_backbone(variant, "vision", model_type=variant, **options)
            )
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )


def create_task(settings, *, state=None):
    return MesoTask(settings, state=state)
