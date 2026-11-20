"""CORE composition and self-contained source-validation experiment state."""

import torch

from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.backbone import Xception
from detectors.xception.checkpoint import initialize_backbone

from .detector import COREDetector
from .trainer import CORETrainer


class CORETask(ImageClassificationTask):
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
                "consistency_rate": 1.0,
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("core"),
                "objective": "cosine_consistency_mse_plus_ce",
                "pairing": "positional_batch_halves_floor_count",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply CORE init_weights or explicitly select initialization: random"
                )
        if construction.get("consistency_rate") != 1.0:
            raise ValueError(
                "CORE checkpoint must preserve the native consistency rate of 1.0"
            )
        options = {k: v for k, v in construction.items() if k != "consistency_rate"}
        if weights is not None:
            with torch.device("meta"):
                model = COREDetector(Xception(**options))
            load_strict(model, weights, assign=True)
        else:
            model = COREDetector(Xception(**options))
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
        report = CORETrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report
