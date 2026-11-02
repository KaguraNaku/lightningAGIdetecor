"""FFD regression recipe on the common Task -> Detector -> Backbone pipeline."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.checkpoint import initialize_backbone
from .detector import FFDDetector
from .trainer import FFDTrainer


CONSTRUCTION = {
    "backbone": {
        "num_classes": 2,
        "in_channels": 3,
        "mode": "original",
        "dropout": 0.0,
    },
    "recipe": {
        "maptype": "reg",
        "map_channels": 728,
        "insertion": "after_block7",
        "with_mask": False,
        "objective": "cross_entropy_only",
    },
}


class FFDTask(ImageClassificationTask):
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
            if settings.get("maptype", "reg") != "reg":
                raise ValueError("Only the configured FFD regression map is connected")
            if settings.get("with_mask", False):
                raise ValueError("FFD source mask-supervised branch is unconnected")
            construction = deepcopy(CONSTRUCTION)
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("ffd"),
                "source_recipe": "ffd.yaml:reg,original,with_maskFalse",
                "objective": "cross_entropy_only",
                "unused_template_io": "Omitted10 unused template reads in regression route; no model/RNG change",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply FFD init_weights or explicitly choose initialization: random"
                )
        if construction != CONSTRUCTION:
            raise ValueError(
                "FFD checkpoint must retain the configured regression recipe"
            )
        if weights is not None:
            with torch.device("meta"):
                model = FFDDetector(
                    load_backbone(
                        "xception",
                        "vision",
                        model_type="xception",
                        **construction["backbone"],
                    )
                )
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
                origin["initialization"] = "explicit_constructed_random"
            model = FFDDetector(backbone)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        result = FFDTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return result


def create_task(settings, *, state=None):
    return FFDTask(settings, state=state)
