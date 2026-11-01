"""SIA task and standalone constructor state on the shared CE fit pipeline."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.efficientnetb4.checkpoint import initialize_backbone

from .detector import SIADetector


ATTENTION = {
    "distance_layout": "concatenate_offsets_then_inner_range_view",
    "distance_gradients": False,
    "normalization": "whole_batch_mean_no_epsilon",
    "channel_range": 5,
    "band_width": 1.0,
    "drop_connect": False,
}


class SIATask(ImageClassificationTask):
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
                    "mode": settings.get("mode", "Original"),
                    "dropout": settings.get("dropout", 0.0),
                },
                "image_size": 256,
                "attention": dict(ATTENTION),
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("sia"),
                "objective": "cross_entropy",
                "recipe": "native_information_attention_three_sites",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply SIA init_weights or explicitly select initialization: random"
                )
        if (
            construction.get("image_size") != 256
            or construction.get("attention") != ATTENTION
        ):
            raise ValueError(
                "SIA state must preserve native geometry and detached batch-global attention"
            )
        options = construction["backbone"]
        if options["num_classes"] != 2 or options["in_channels"] != 3:
            raise ValueError("SIA requires an RGB binary-classification backbone")

        def build_backbone():
            return load_backbone(
                "efficientnetb4", "vision", model_type="efficientnetb4", **options
            )

        if weights is not None:
            with torch.device("meta"):
                model = SIADetector(build_backbone())
            load_strict(model, weights, assign=True)
        else:
            backbone = build_backbone()
            if settings.get("init_weights"):
                origin["pretrained_backbone"] = initialize_backbone(
                    backbone, settings["init_weights"]
                )
            else:
                origin["initialization"] = "explicit_random"
            model = SIADetector(backbone)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )


def create_task(settings, *, state=None):
    return SIATask(settings, state=state)
