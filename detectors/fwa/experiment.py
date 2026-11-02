"""FWA Task -> paired generation/trainer -> Detector -> lazy Backbone."""

from copy import deepcopy
from pathlib import Path

import torch

from backbones.load_backbone import load_backbone
from data.manifest import sha256
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from detectors.xception.checkpoint import initialize_backbone
from .detector import FWADetector
from .generation import FWAPairCollator
from .trainer import FWATrainer

CONSTRUCTION = {
    "backbone": {
        "num_classes": 2,
        "in_channels": 3,
        "mode": "original",
        "dropout": 0.0,
    },
    "generation_recipe": {
        "id": "local-fwa-generated-pairs-v1",
        "resolution": 256,
        "predictor": "dlib20_first68_of81",
        "similarity": "source_U_diag_d_V_transpose",
        "scale_choices": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "blur": [5, 5],
        "mask_deform_std": 20,
        "source_augmentations_unconditional": True,
        "training_channel_swap": True,
        "discarded_post_augmentation_rng_retained": True,
        "collate": "all_generated_then_all_original_then_shuffle",
        "fallback": "duplicate_augmented_original_with_source_label",
    },
}


class FWATask(ImageClassificationTask):
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
            if not isinstance(generation, dict) or set(generation) != {"predictor"}:
                raise ValueError(
                    "FWA checkpoint requires explicit training generation identity"
                )
            self.predictor = deepcopy(generation["predictor"])
            # Inference restoration must not read a training-only external asset.
            if self.predictor is not None and settings.get("predictor_path"):
                self.predictor["path"] = str(
                    Path(settings["predictor_path"]).expanduser().resolve()
                )
        else:
            construction = deepcopy(CONSTRUCTION)
            preprocessing = preprocessing_config(
                256, (0.5,) * 3, (0.5,) * 3, augmentation=False
            )
            self.predictor = None
            if settings.get("predictor_path"):
                path = Path(settings["predictor_path"]).expanduser().resolve()
                self.predictor = {
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            origin = {
                **reference_for("fwa"),
                "objective": "native_full_2B_cross_entropy",
                "source_label_transport": "Explicit manifest real0/fake1 replaces source absolute-path segment6 inference",
                "data_generation": "native_fwa_with_discarded_rng_and_no_face_fallback",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = read_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "Supply FWA init_weights or explicitly choose initialization: random"
                )
        if construction != CONSTRUCTION:
            raise ValueError(
                "FWA checkpoint must retain the complete native pair recipe"
            )
        if self.predictor is not None and (
            not isinstance(self.predictor, dict)
            or set(self.predictor) != {"path", "sha256", "bytes"}
        ):
            raise ValueError("FWA predictor identity must retain path/hash/size")
        if weights is not None:
            with torch.device("meta"):
                model = FWADetector(
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
            model = FWADetector(backbone)
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def get_collate_fn(self, stage):
        return (
            FWAPairCollator(self.predictor)
            if stage == "train"
            else super().get_collate_fn(stage)
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        result = FWATrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return result

    def state_dict(self):
        return {
            **super().state_dict(),
            "training_generation": {"predictor": deepcopy(self.predictor)},
        }


def create_task(settings, *, state=None):
    return FWATask(settings, state=state)
