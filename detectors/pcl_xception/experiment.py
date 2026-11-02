"""PCL Task -> pinned I2G sources / mask trainer -> SLADD-Xception detector."""

from copy import deepcopy
from pathlib import Path

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .bank import bank_identity
from .checkpoint import initialize_backbone
from .dataset import I2GDataset, I2GPairCollator
from .detector import PCLDetector
from .trainer import PCLTrainer


CONSTRUCTION = {
    "backbone": "native_xception_sladd_all_unused_heads",
    "recipe": {
        "resolution": 256,
        "normalization": [0.5, 0.5],
        "num_classes": 2,
        "correlation": "theta_twice_unused_phi_728_sqrt_sigmoid_transpose",
        "target": "bilinear_align_cornersTrue_16_over256_raw_first_width_column_each_row",
        "objective": "CE+1*BCE_sigmoid_map",
        "mask": "raw_float32_alpha_not_boundary",
        "generation": "native_I2G_clip_reorder_four_hulls_ratio1_no_source_aug_no_hflip",
        "backgrounds": "real0_deduplicated_storage_stable_first_manifest_order",
        "view_order": "real_all_then_fake_all_no_shuffle",
        "decode_failure": "first_source_replacement_once_then_fatal",
        "scheduler": "StepLR3_gamma0.4_once_after_complete_fit",
    },
}


class PCLTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        expected_preprocessing = preprocessing_config(
            256, (0.5,) * 3, (0.5,) * 3, augmentation=False
        )
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
            generation, training = (
                state.get("training_generation"),
                state.get("training_state"),
            )
            if (
                not isinstance(generation, dict)
                or set(generation) != {"bank"}
                or not isinstance(training, dict)
                or set(training) != {"scheduler"}
            ):
                raise ValueError(
                    "PCL state requires explicit training bank and scheduler state"
                )
            self.bank, self.scheduler_state = (
                deepcopy(generation["bank"]),
                deepcopy(training["scheduler"]),
            )
            if self.bank is not None and settings.get("bank_path"):
                self.bank["path"] = str(
                    Path(settings["bank_path"]).expanduser().resolve()
                )
        else:
            construction, preprocessing = deepcopy(CONSTRUCTION), expected_preprocessing
            self.bank = (
                bank_identity(settings["bank_path"])
                if settings.get("bank_path")
                else None
            )
            self.scheduler_state = None
            origin = {
                **reference_for("pcl_xception"),
                "initialization": "native234_ImageNet_keys_with_fresh_classifier_auxiliary_heads_and_PCL",
                "selection_transport": "Explicit train manifest and ordered neighbor bank; stable dedup replaces native hash-set order",
                "failure_boundary": "Native first-source replacement retained once, persistent failure stops explicitly; nonfinite raw generation fatal",
                "exact_optimizer_rng_resume_supported": False,
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply original Xception init_weights or explicitly select initialization: random"
                )
        if construction != CONSTRUCTION or preprocessing != expected_preprocessing:
            raise ValueError(
                "PCL requires its complete fixed native model/preprocessing/I2G recipe"
            )
        if self.bank is not None and (
            not isinstance(self.bank, dict)
            or set(self.bank) != {"path", "sha256", "bytes"}
        ):
            raise ValueError("PCL training bank identity incomplete")

        def build():
            return PCLDetector(
                load_backbone("xception_sladd", "vision", model_type="xception_sladd")
            )

        if weights is not None:
            with torch.device("meta"):
                model = build()
            load_strict(model, weights, assign=True)
        else:
            model = build()
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
        if stage != "train":
            return dataset
        if self.bank is None:
            raise ValueError("PCL training requires explicit prepared bank_path")
        return I2GDataset(dataset, self.bank)

    def get_collate_fn(self, stage):
        return I2GPairCollator() if stage == "train" else super().get_collate_fn(stage)

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        trainer = PCLTrainer(
            self.model, origin=self.origin, scheduler_state=self.scheduler_state
        )
        report = trainer.fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self.scheduler_state, self._ready = trainer.scheduler_state, True
        return report

    def state_dict(self):
        return {
            **super().state_dict(),
            "training_generation": {"bank": deepcopy(self.bank)},
            "training_state": {"scheduler": deepcopy(self.scheduler_state)},
        }


def create_task(settings, *, state=None):
    return PCLTask(settings, state=state)
