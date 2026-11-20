"""F3Net FAD composition with preserved constructor RNG and standalone state."""

from detectors._shared.checkpoint import restore_model
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.backbone import Xception
from detectors.xception.checkpoint import read_initial_detector_weights

from .checkpoint import initialize_backbone, replace_stem
from .detector import F3NetDetector


class F3NetTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        if state is not None:
            construction, preprocessing, origin, weights = self.unpack_state(state)
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
            weights, receipt = read_initial_detector_weights(settings, detector="F3Net")
            origin = {
                **reference_for("f3net"),
                "objective": "cross_entropy",
                "recipe": "local_FAD_only_full_Xception",
                **receipt,
            }
        if (
            construction.get("stem_channels") != 12
            or construction["backbone"]["in_channels"] != 3
        ):
            raise ValueError(
                "F3Net requires RGB backbone construction before twelve-channel stem replacement"
            )

        def build(init_weights=None):
            backbone = Xception(**construction["backbone"])
            if init_weights:
                origin["pretrained_backbone"] = initialize_backbone(backbone, init_weights)
            else:
                replace_stem(backbone)
            return F3NetDetector(backbone, construction["image_size"])

        if weights is not None:
            model = restore_model(build, weights)
        else:
            init_weights = settings.get("init_weights")
            model = build(init_weights)
            if not init_weights:
                origin["initialization"] = "explicit_random"
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )
