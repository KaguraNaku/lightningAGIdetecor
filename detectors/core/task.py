"""CORE composition and self-contained source-validation experiment state."""

from detectors._shared.checkpoint import restore_model
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.backbone import Xception
from detectors.xception.checkpoint import initialize_backbone, read_initial_detector_weights

from .detector import COREDetector
from .trainer import CORETrainer


class CORETask(ImageClassificationTask):
    trainer_class = CORETrainer

    def __init__(self, settings, *, state=None):
        if state is not None:
            construction, preprocessing, origin, weights = self.unpack_state(state)
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
            weights, receipt = read_initial_detector_weights(settings, detector="CORE")
            origin = {
                "method": "core",
                "objective": "cosine_consistency_mse_plus_ce",
                "pairing": "positional_batch_halves_floor_count",
                **receipt,
            }
        if construction.get("consistency_rate") != 1.0:
            raise ValueError(
                "CORE checkpoint must preserve the native consistency rate of 1.0"
            )
        options = {k: v for k, v in construction.items() if k != "consistency_rate"}

        def build():
            return COREDetector(Xception(**options))

        if weights is not None:
            model = restore_model(build, weights)
        else:
            model = build()
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
