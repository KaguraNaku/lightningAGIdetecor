"""Xception source-validation experiment task."""

from detectors._shared.checkpoint import restore_model
from detectors._shared.detector import BackboneClassifier
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config

from .backbone import Xception
from .checkpoint import initialize_backbone, read_initial_detector_weights


class XceptionTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        if state is not None:
            construction, preprocessing, origin, weights = self.unpack_state(state)
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
            weights, receipt = read_initial_detector_weights(settings, detector="Xception")
            origin = {"method": "xception", "objective": "cross_entropy", **receipt}

        def build():
            return BackboneClassifier(Xception(**construction))

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
