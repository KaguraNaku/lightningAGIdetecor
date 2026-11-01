"""SPSL composition on the shared image-classification pipeline."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

from .checkpoint import initialize_backbone
from .detector import SPSLDetector


class SPSLTask(ImageClassificationTask):
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
                "in_channels": 4,
                "mode": settings.get("mode", "original"),
                "dropout": settings.get("dropout", 0.0),
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("spsl"),
                "objective": "cross_entropy",
                "phase": "mean_RGB_fft_angle_unit_spectrum_inverse_real",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply SPSL init_weights or explicitly select initialization: random"
                )
        if construction["in_channels"] != 4:
            raise ValueError(
                "SPSL checkpoint must retain the native four-channel input"
            )
        if weights is not None:
            with torch.device("meta"):
                model = SPSLDetector(
                    load_backbone(
                        "xception", "vision", model_type="xception", **construction
                    )
                )
            load_strict(model, weights, assign=True)
        else:
            model = SPSLDetector(
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
    return SPSLTask(settings, state=state)
