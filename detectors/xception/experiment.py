"""Xception source-validation experiment task."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.detector import BackboneClassifier
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors._shared.provenance import reference_for

from .checkpoint import initialize_backbone


class XceptionTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        origin = {**reference_for("xception"), "objective": "cross_entropy"}
        weights = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
            )
            weights = state["state_dict"]
        else:
            construction = {
                "num_classes": 2,
                "in_channels": 3,
                "mode": settings.get("mode", "original"),
                "dropout": settings.get("dropout", 0.0),
            }
            preprocessing = preprocessing_config(
                settings.get("image_size", 256),
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", True),
                data_aug={"rotate_prob": 0.0, **settings.get("data_aug", {})},
            )
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply Xception init_weights or explicitly select initialization: random"
                )
        if weights is not None:
            with torch.device("meta"):
                model = BackboneClassifier(
                    load_backbone(
                        "xception", "vision", model_type="xception", **construction
                    )
                )
            load_strict(model, weights, assign=True)
        else:
            model = BackboneClassifier(
                load_backbone(
                    "xception", "vision", model_type="xception", **construction
                )
            )
            if settings.get("init_weights"):
                origin["pretrained_backbone"] = initialize_backbone(
                    model.backbone, settings["init_weights"]
                )
            else:
                origin["initialization"] = "explicit_random"
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )


def create_task(settings, *, state=None):
    return XceptionTask(settings, state=state)
