"""FaceXray Task -> mapped source bank / paired mask trainer -> HRNet detector."""

from copy import deepcopy
from pathlib import Path

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import read_weights, load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .bank import bank_identity
from .checkpoint import initialize_backbone
from .dataset import FaceXrayDataset, FaceXrayPairCollator
from .detector import FaceXrayDetector
from .trainer import FaceXrayTrainer


CONSTRUCTION = {
    "backbone": "native_hrnet_w48_full_unused_classification_state",
    "recipe": {
        "resolution": 256,
        "feature_channels": 720,
        "backbone_bilinear_align_corners": False,
        "boundary_bilinear_align_corners": True,
        "boundary": "4*blurred_alpha*(1-alpha)_float64_then_source_float32_MSE",
        "objective": "cross_entropy+1000*raw_boundary_MSE_no_sigmoid",
        "pool": [128, 128],
        "classifier": [16384, 1024, 128, 2],
        "train_backgrounds": "all_explicit_manifest_real0_and_fake1",
        "neighbors": "ordered_python_choice_or_self_if_key_absent_no_silent_filter",
        "mask": "native_five_hulls_bg_deform_then_second_std20_deform",
        "warp": "random_five_point_affine_or_Delaunay_first48",
        "training_channel_swap": True,
        "discarded_post_augmentation_rng_retained": True,
        "order": "fake_all_then_real_all_joint_image_mask_label_shuffle",
        "scheduler": None,
    },
}


class FaceXrayTask(ImageClassificationTask):
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
            generation = state.get("training_generation")
            if not isinstance(generation, dict) or set(generation) != {"bank"}:
                raise ValueError(
                    "FaceXray state requires explicit training bank identity"
                )
            self.bank = deepcopy(generation["bank"])
            if self.bank is not None and settings.get("bank_path"):
                self.bank["path"] = str(
                    Path(settings["bank_path"]).expanduser().resolve()
                )
        else:
            construction = deepcopy(CONSTRUCTION)
            preprocessing = preprocessing_config(
                256, (0.5,) * 3, (0.5,) * 3, augmentation=False
            )
            self.bank = (
                bank_identity(settings["bank_path"])
                if settings.get("bank_path")
                else None
            )
            origin = {
                **reference_for("facexray"),
                "objective": "CE+1000*raw_mask_MSE_source_float32",
                "source_initialization": "HRNet1954_matches_ignore_classifier2_retain_fresh_unused_fc2",
                "label_transport": "Explicit manifest labels replace source path-component6 guessing; no real-only source filter",
                "neighbor_transport": "Complete compiled train inventory; provided lists preserved or separately declared train-scoped KDTree preparation",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply original HRNet init_weights or explicitly select initialization: random"
                )
        if construction != CONSTRUCTION:
            raise ValueError(
                "FaceXray requires its complete fixed native HRNet/boundary/pair recipe"
            )
        if self.bank is not None and (
            not isinstance(self.bank, dict)
            or set(self.bank) != {"path", "sha256", "bytes"}
        ):
            raise ValueError("FaceXray training bank identity is incomplete")

        def build():
            return FaceXrayDetector(
                load_backbone("hrnet_w48", "vision", model_type="hrnet_w48")
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
            raise ValueError(
                "FaceXray training requires an explicit prepared bank_path"
            )
        return FaceXrayDataset(dataset, self.bank)

    def get_collate_fn(self, stage):
        return (
            FaceXrayPairCollator()
            if stage == "train"
            else super().get_collate_fn(stage)
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = FaceXrayTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report

    def state_dict(self):
        return {
            **super().state_dict(),
            "training_generation": {"bank": deepcopy(self.bank)},
        }


def create_task(settings, *, state=None):
    return FaceXrayTask(settings, state=state)
