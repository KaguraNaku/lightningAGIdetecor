"""RFM composition with explicit local-source recipe and standalone task state."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.checkpoint import initialize_backbone

from .detector import RFMDetector
from .trainer import RFMTrainer

RECIPE = "local_feature_difference_with_parameter_gradients_v1"


class RFMTask(ImageClassificationTask):
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
                "mode": settings.get("mode", "original"),
                "dropout": settings.get("dropout", 0.0),
                "rfm_recipe": RECIPE,
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", True),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("rfm"),
                "objective": "CE_with_retained_native_FAM_gradients",
                "rfm_target": "activated_feature_channel_1_minus_0_spatial_sum",
                "rfm_recipe": RECIPE,
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply RFM init_weights or explicitly select initialization: random"
                )
        if construction.get("rfm_recipe") != RECIPE:
            raise ValueError("Unsupported RFM checkpoint recipe")
        options = {k: v for k, v in construction.items() if k != "rfm_recipe"}
        if weights is not None:
            with torch.device("meta"):
                model = RFMDetector(
                    load_backbone(
                        "xception", "vision", model_type="xception", **options
                    )
                )
            load_strict(model, weights, assign=True)
        else:
            model = RFMDetector(
                load_backbone("xception", "vision", model_type="xception", **options)
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

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = RFMTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report


def create_task(settings, *, state=None):
    return RFMTask(settings, state=state)
