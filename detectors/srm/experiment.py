"""SRM task composition, independent branch initialization and standalone state."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.checkpoint import initialize_backbone

from .detector import SRMDetector
from .trainer import SRMTrainer


OBJECTIVE = {
    "margin_type": "cos",
    "gamma": 0.0,
    "m": 0.45,
    "s": 30,
    "t": 1.0,
    "input": "raw_logits",
}


class SRMTask(ImageClassificationTask):
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
                    "dropout": settings.get("dropout", 0.0),
                },
                "image_size": 256,
                "attention_size": 16,
                "objective": dict(OBJECTIVE),
            }
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("srm"),
                "objective": dict(OBJECTIVE),
                "recipe": "dual_Xception_fixed_SRM_cross_attention",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply SRM init_weights or explicitly select initialization: random"
                )
        if (
            construction.get("image_size") != 256
            or construction.get("attention_size") != 16
            or construction.get("objective") != OBJECTIVE
        ):
            raise ValueError(
                "SRM state must preserve fixed geometry and raw-logit margin objective"
            )
        options = construction["backbone"]
        if options["in_channels"] != 3 or options["num_classes"] != 2:
            raise ValueError("SRM requires two RGB binary-classification backbones")

        def build_branch():
            return load_backbone("xception", "vision", model_type="xception", **options)

        if weights is not None:
            with torch.device("meta"):
                model = SRMDetector(build_branch(), build_branch())
            load_strict(model, weights, assign=True)
        else:
            # Loading each branch before constructing the next mirrors the source.
            branches, receipts = [], []
            for _ in range(2):
                branch = build_branch()
                if settings.get("init_weights"):
                    receipts.append(
                        initialize_backbone(branch, settings["init_weights"])
                    )
                branches.append(branch)
            model = SRMDetector(*branches)
            if receipts:
                origin["pretrained_backbones"] = receipts
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
        report = SRMTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report


def create_task(settings, *, state=None):
    return SRMTask(settings, state=state)
