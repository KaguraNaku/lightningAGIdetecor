"""STIL Task composition over its native detector and explicit local weights."""

from copy import deepcopy

import torch

from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from .checkpoint import initialize_detector, restore_detector, trained_weights
from .dataset import STILDataset, STILCollator
from .trainer import STILTrainer


RECIPE = {
    "backbone": "native_SCNet50_v1d_T8",
    "clip_shape": [8, 3, 224, 224],
    "frame_pool": "all_declared_source_frames_then_native_first32_T8_selection",
    "head": "SCNet1000_pretraining_then_fresh_Linear2048to2",
    "consensus": "mean_eight_frame_logits_before_class1_softmax",
    "objective": "two_logit_cross_entropy",
    "prediction_unit": "clip",
    "native_batch_size": 8,
    "scheduler": None,
    "precision_scope": "fp32_alignment_profile_not_native_CUDA_AMP_reproduction",
}
PREPROCESSING = {
    "resolution": 224,
    "clip_size": 8,
    "channels": "RGB",
    "resize": "OpenCV_INTER_CUBIC",
    "mean": [0.5] * 3,
    "std": [0.5] * 3,
    "augmentation": "native_Albu1.3.1_shared_clip_seed",
    "train_augmented": True,
    "validation_augmented": True,
    "evaluation_augmented": False,
    "worker_RNG": "restore_Python_NumPy_state_instead_of_native_entropy_reset",
    "decode_failure": "fatal_one_attempt_no_clip_video_or_label_substitution",
    "with_mask": False,
    "with_landmark": False,
}


class STILTask(ImageClassificationTask):
    prediction_unit = "clip"

    def __init__(self, settings, *, state=None):
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("The selected native STIL profile requires float32")
        if settings.get("score_type", "probability") != "probability":
            raise ValueError("Native STIL returns a class1 probability")
        if state is not None:
            self.validate_state(state)
            construction = deepcopy(state["construction"])
            preprocessing = deepcopy(state["preprocessing"])
            origin = deepcopy(state["origin"])
            weights = state["state_dict"]
        else:
            construction = {
                "recipe": deepcopy(RECIPE),
                "prediction_unit": "clip",
                "selection_seed": settings.get("selection_seed", 1024),
            }
            preprocessing = deepcopy(PREPROCESSING)
            origin = {
                **reference_for("stil"),
                "selection_policy": "Remake source train/val only;no target-set checkpoint selection",
                "source_pool": "Declared all-frame manifest with hash-bound files-first metadata;not author-corpus discovery",
                "native_process_RNG_reproduced": False,
                "exact_optimizer_rng_resume_supported": False,
                "full_experiment_reproduction": False,
                "native_generic_Trainer_use_amp_default": True,
                "GPU_AMP_execution_verified": False,
                "native_yaml_backbone_name": "Stale xception field is unused;actual architecture is SCNet50_v1d",
            }
            weights = None
            if settings.get("initial_checkpoint"):
                if settings.get("native_override"):
                    raise ValueError(
                        "Full trained STIL import and two-stage initialization are separate operations"
                    )
                weights, receipt = trained_weights(settings["initial_checkpoint"])
                origin.update(receipt)
                origin["initialization_precedence"] = (
                    "Complete Detector import skips initialization-only SCNet path;"
                    "compatible with the shared checkpoint conversion CLI"
                )
        self.validate_metadata(construction, preprocessing, origin)
        if state is not None or weights is not None:
            model = restore_detector(weights)
        else:
            model, receipt = initialize_detector(
                settings.get("init_weights"),
                native_override=settings.get("native_override"),
            )
            origin["initialization"] = receipt
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=state is not None or weights is not None,
        )

    @staticmethod
    def validate_state(state):
        if (
            not isinstance(state, dict)
            or set(state)
            != {"version", "construction", "preprocessing", "origin", "state_dict"}
            or type(state["version"]) is not int
            or state["version"] != 1
        ):
            raise ValueError("Unsupported or incomplete STIL Task state")

    @staticmethod
    def validate_metadata(construction, preprocessing, origin):
        if (
            not isinstance(construction, dict)
            or set(construction) != {"recipe", "prediction_unit", "selection_seed"}
            or construction["recipe"] != RECIPE
            or construction["prediction_unit"] != "clip"
            or type(construction["selection_seed"]) is not int
            or not 0 <= construction["selection_seed"] < 2**32
            or preprocessing != PREPROCESSING
            or not isinstance(origin, dict)
        ):
            raise ValueError("STIL requires its complete native T8 recipe and roles")

    def get_collate_fn(self, stage):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown STIL preprocessing stage")
        return STILCollator()

    def prepare_dataset(self, dataset, *, stage):
        return STILDataset(
            dataset, stage=stage, seed=self.construction["selection_seed"]
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._ready = False
        report = STILTrainer(self.model, origin=self.origin).fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings=settings,
            output_dir=output_dir,
        )
        self._ready = True
        return report

    @torch.inference_mode()
    def predict(self, clips, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import complete STIL weights before prediction")
        if runtime.precision != "fp32":
            raise ValueError("The selected native STIL alignment profile requires fp32")
        self.model.to(runtime.device).eval()
        scores = self.model.forward_outputs(
            clips.to(runtime.device, non_blocking=True)
        )["prob"]
        if (
            scores.shape != (len(clips),)
            or not torch.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
        ):
            raise ValueError("STIL must return one finite fake probability per clip")
        return scores


def create_task(settings, *, state=None):
    return STILTask(settings, state=state)
