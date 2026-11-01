"""RECCE task preserves the unused wrapper branch and distinct active encoder."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

from .detector import RECCEDetector


RECIPE = {
    "dropout": 0.2,
    "graph_ratios": [2, 4],
    "objective": "cross_entropy_only",
    "training_noise_std": 1e-6,
    "auxiliary_observations": ["reconstruction", "correlations"],
}


class RECCETask(ImageClassificationTask):
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
                    "mode": "original",
                    "dropout": 0.0,
                },
                "encoder": {"num_classes": 1000, "in_channels": 3},
                "image_size": 256,
                "recipe": dict(RECIPE),
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("recce"),
                "objective": "cross_entropy_only_local_wrapper",
                "encoder_identity": "legacy_xception.tf_in1k",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply RECCE active encoder init_weights or explicitly select initialization: random"
                )
        if (
            construction.get("image_size") != 256
            or construction.get("recipe") != RECIPE
            or construction.get("encoder") != {"num_classes": 1000, "in_channels": 3}
            or construction.get("backbone")
            != {"num_classes": 2, "in_channels": 3, "mode": "original", "dropout": 0.0}
        ):
            raise ValueError(
                "RECCE state must preserve native branches, geometry and CE-only recipe"
            )

        def branches():
            unused = load_backbone(
                "xception", "vision", model_type="xception", **construction["backbone"]
            )
            encoder = load_backbone(
                "xception_timm",
                "vision",
                model_type="xception_timm",
                **construction["encoder"],
            )
            return unused, encoder

        if weights is not None:
            with torch.device("meta"):
                model = RECCEDetector(*branches())
            load_strict(model, weights, assign=True)
        else:
            unused, encoder = branches()
            if settings.get("init_weights"):
                initialized, receipt = read_weights(settings["init_weights"])
                encoder.load_state_dict(initialized, strict=True)
                origin["pretrained_encoder"] = receipt
            else:
                origin["initialization"] = "explicit_random"
            model = RECCEDetector(unused, encoder)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )


def create_task(settings, *, state=None):
    return RECCETask(settings, state=state)
