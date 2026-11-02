"""CapsuleNet Task composes its independent VGG prefix and native trainer."""

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .detector import CapsuleNetDetector
from .trainer import CapsuleTrainer


RECIPE = {
    "image_size": 256,
    "num_classes": 2,
    "capsules": 10,
    "data_in": 8,
    "data_out": 4,
    "routing_iterations": 2,
    "vgg_end_inclusive": 18,
    "vgg_trainable": True,
    "routing_random": False,
    "routing_dropout": 0.0,
    "loss": "sum_four_probability_input_cross_entropy",
    "score": "softmax_of_detached_mean_capsule_probabilities",
}


class CapsuleNetTask(ImageClassificationTask):
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
            construction = dict(RECIPE)
            preprocessing = preprocessing_config(
                256,
                (0.5,) * 3,
                (0.5,) * 3,
                augmentation=settings.get("augmentation", False),
                data_aug=settings.get("data_aug"),
            )
            origin = {
                **reference_for("capsule_net"),
                "objective": RECIPE["loss"],
                "score": RECIPE["score"],
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply CapsuleNet init_weights or explicitly select initialization: random"
                )
        if construction != RECIPE:
            raise ValueError("CapsuleNet state must preserve its native recipe")

        def backbone(**options):
            return load_backbone(
                "vgg19_capsule", "vision", model_type="vgg19_capsule", **options
            )

        if weights is not None:
            with torch.device("meta"):
                model = CapsuleNetDetector(backbone())
            load_strict(model, weights, assign=True)
        else:
            pretrained = None
            if settings.get("init_weights"):
                pretrained, receipt = read_weights(settings["init_weights"])
                origin["pretrained_backbone"] = receipt
            else:
                origin["initialization"] = (
                    "explicit_constructed_random_pretrained_layout"
                )
            model = CapsuleNetDetector(backbone(weights=pretrained))
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = CapsuleTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={"epochs": 11, **settings},
            output_dir=output_dir,
        )
        self._ready = True
        return report


def create_task(settings, *, state=None):
    return CapsuleNetTask(settings, state=state)
