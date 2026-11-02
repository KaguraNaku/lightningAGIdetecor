"""SBI Task -> real-only dataset / ordered pair trainer -> three-logit B4."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import read_weights, load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.efficientnetb4.checkpoint import initialize_backbone
from .dataset import SBIRealDataset
from .detector import SBIDetector
from .generation import SBIPairCollator
from .trainer import SBITrainer

CONSTRUCTION = {
    "backbone": {
        "num_classes": 3,
        "in_channels": 3,
        "mode": "original",
        "dropout": 0.0,
    },
    "recipe": {
        "source_filter": "real0_before_sampling",
        "source_coordinates": 256,
        "landmarks": "fingerprinted_manifest_or_native_frames_npy_68_or81",
        "hulls": [0, 1, 2, 3],
        "random_truncate68_probability": 0.25,
        "synchronized_augmentation": True,
        "source_channel_swap": False,
        "elastic_image_discarded_mask_retained": True,
        "dynamic_blend_ratios": [0.25, 0.5, 0.75, 1, 1, 1],
        "order": "all_real_then_all_fake_no_shuffle",
        "fake_score": "softmax_all3_class1",
        "objective": "cross_entropy_all3",
        "scheduler": "StepLR3_gamma0.4_once_after_complete_fit",
    },
}


class SBITask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        self.scheduler_state = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
            method = state.get("method_state")
            if not isinstance(method, dict) or set(method) != {"scheduler"}:
                raise ValueError("SBI state requires explicit scheduler carrier")
            self.scheduler_state = method["scheduler"]
            if self.scheduler_state is not None and not isinstance(
                self.scheduler_state, dict
            ):
                raise ValueError("SBI scheduler state must be a dictionary or None")
        else:
            construction = deepcopy(CONSTRUCTION)
            preprocessing = preprocessing_config(
                256, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), augmentation=False
            )
            origin = {
                **reference_for("sbi"),
                "objective": "CE_all3_logits_labels0or1",
                "source_initialization": "B4_internal_pretrained_then_fresh_stem_and3_head",
                "unused_outer_xception_import": "Omitted: native strictFalse matches0 tensor keys and changes no state/RNG",
                "failure_visibility": "Surface native fatal generator/dataset failure with original cause and RNG; no replacement image",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply SBI B4 init_weights or explicitly select initialization: random; source outer Xception file is unused"
                )
        if construction != CONSTRUCTION:
            raise ValueError(
                "SBI state must retain the complete native three-logit/pair recipe"
            )
        if weights is not None:
            with torch.device("meta"):
                model = SBIDetector(
                    load_backbone(
                        "efficientnetb4",
                        "vision",
                        model_type="efficientnetb4",
                        **construction["backbone"],
                    )
                )
            load_strict(model, weights, assign=True)
        else:
            model = SBIDetector(
                load_backbone(
                    "efficientnetb4",
                    "vision",
                    model_type="efficientnetb4",
                    **construction["backbone"],
                )
            )
            if settings.get("init_weights"):
                origin["pretrained_backbone"] = initialize_backbone(
                    model.backbone, settings["init_weights"]
                )
            else:
                origin["initialization"] = "explicit_constructed_random"
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def prepare_dataset(self, dataset, *, stage):
        return SBIRealDataset(dataset) if stage == "train" else dataset

    def get_collate_fn(self, stage):
        return SBIPairCollator() if stage == "train" else super().get_collate_fn(stage)

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import SBI weights before prediction")
        self.model.to(runtime.device).eval()
        with runtime.autocast():
            logits = self.model(images.to(runtime.device, non_blocking=True))
            if logits.shape != (len(images), 3):
                raise ValueError("SBI inference retains all three logits")
            return logits.softmax(1)[:, 1]

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        trainer = SBITrainer(
            self.model, origin=self.origin, scheduler_state=self.scheduler_state
        )
        result = trainer.fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self.scheduler_state = trainer.scheduler_state
        self._ready = True
        return result

    def state_dict(self):
        return {
            **super().state_dict(),
            "method_state": {"scheduler": deepcopy(self.scheduler_state)},
        }


def create_task(settings, *, state=None):
    return SBITask(settings, state=state)
