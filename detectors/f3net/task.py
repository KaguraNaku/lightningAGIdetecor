"""F3Net FAD composition with preserved constructor RNG and standalone state."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

from .checkpoint import initialize_backbone, replace_stem
from .detector import F3NetDetector


class F3NetTask(ImageClassificationTask):
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
                "backbone": {
                    "num_classes": 2,
                    "in_channels": 3,
                    "mode": settings.get("mode", "original"),
                    "dropout": settings.get("dropout", 0.5),
                },
                "image_size": settings.get("image_size", 256),
                "stem_channels": 12,
            }
            preprocessing = preprocessing_config(
                construction["image_size"],
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("f3net"),
                "objective": "cross_entropy",
                "recipe": "local_FAD_only_full_Xception",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply F3Net init_weights or explicitly select initialization: random"
                )
        if (
            construction.get("stem_channels") != 12
            or construction["backbone"]["in_channels"] != 3
        ):
            raise ValueError(
                "F3Net requires RGB backbone construction before twelve-channel stem replacement"
            )
        if weights is not None:
            with torch.device("meta"):
                backbone = load_backbone(
                    "xception",
                    "vision",
                    model_type="xception",
                    **construction["backbone"],
                )
                replace_stem(backbone)
                model = F3NetDetector(backbone, construction["image_size"])
            load_strict(model, weights, assign=True)
        else:
            backbone = load_backbone(
                "xception", "vision", model_type="xception", **construction["backbone"]
            )
            if settings.get("init_weights"):
                origin["pretrained_backbone"] = initialize_backbone(
                    backbone, settings["init_weights"]
                )
            else:
                replace_stem(backbone)
                origin["initialization"] = "explicit_random"
            model = F3NetDetector(backbone, construction["image_size"])
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )
