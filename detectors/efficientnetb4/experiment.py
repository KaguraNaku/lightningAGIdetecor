"""Native B4 composition on the shared source-validation image task."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.detector import BackboneClassifier
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

from .checkpoint import initialize_backbone


class EfficientNetB4Task(ImageClassificationTask):
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
            construction = {
                "num_classes": 2,
                "in_channels": 3,
                "mode": settings.get("mode", "Original"),
                "dropout": settings.get("dropout", 0.0),
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
            origin = {**reference_for("efficientnetb4"), "objective": "cross_entropy"}
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply EfficientNet-B4 init_weights or explicitly select initialization: random"
                )
        if weights is not None:
            with torch.device("meta"):
                model = BackboneClassifier(
                    load_backbone(
                        "efficientnetb4",
                        "vision",
                        model_type="efficientnetb4",
                        **construction,
                    )
                )
            load_strict(model, weights, assign=True)
        else:
            model = BackboneClassifier(
                load_backbone(
                    "efficientnetb4",
                    "vision",
                    model_type="efficientnetb4",
                    **construction,
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
    return EfficientNetB4Task(settings, state=state)
