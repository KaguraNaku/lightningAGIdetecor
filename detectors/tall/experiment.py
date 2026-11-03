"""TALL composition with an explicitly selected native data recipe."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from .checkpoint import require_profile, restore_detector, trained_weights
from .dataset import TALLCollator, TALLDataset
from .detector import TALLDetector
from .trainer import TALLTrainer
from .transforms import PROFILES

RECIPE = {
    "backbone": "native_TALL_Swin_base128_depths2_2_18_2",
    "clip_shape": [4, 3, 224, 224],
    "frame_pool": "all_declared_source_frames_then_native_first32_T4_selection",
    "thumbnail": "2_by_2_frames_before_native_bilinear224_resize",
    "absolute_position": "four_frame_embeddings_in56_by56_quadrants",
    "objective": "two_logit_cross_entropy",
    "score": "class1_softmax",
    "prediction_unit": "clip",
    "native_batch_size": 32,
    "all_parameter_tensors_trainable": 330,
    "native_inclusive_epochs": 101,
    "native_optimizer": "Adam_lr2e-5_weight_decay5e-4",
    "scheduler": None,
    "precision_scope": "fp32_alignment_profile_not_native_CUDA_AMP_reproduction",
}


def preprocessing_for(profile):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError(
            "Select TALL native_entrypoint or specialized_cutout explicitly"
        )
    roles = PROFILES[profile]
    return {
        "data_profile": profile,
        "resolution": 224,
        "clip_size": 4,
        "channels": "RGB",
        "resize": "OpenCV_INTER_CUBIC",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "augmentation": "native_Albu1.3.1_shared_clip_seed",
        "train_augmented": "train" in roles["augmentation_stages"],
        "validation_augmented": "validation" in roles["augmentation_stages"],
        "evaluation_augmented": False,
        "cutout_in_all_stages": roles["cutout"],
        "cutout": {"size": 16, "fill_after_normalization": -1, "shared_across_T4": True}
        if roles["cutout"]
        else None,
        "worker_RNG": "restore_Python_NumPy_state_instead_of_native_entropy_reset",
        "decode_failure": "fatal_one_attempt_no_clip_video_or_label_substitution",
        "with_mask": False,
        "with_landmark": False,
    }


class TALLTask(ImageClassificationTask):
    prediction_unit = "clip"

    def __init__(self, settings, *, state=None):
        require_profile()
        if settings.get("score_type", "probability") != "probability":
            raise ValueError("Native TALL returns a class1 probability")
        if state is not None:
            self.validate_state(state)
            construction = deepcopy(state["construction"])
            preprocessing = deepcopy(state["preprocessing"])
            origin = deepcopy(state["origin"])
            weights = state["state_dict"]
        else:
            profile = settings.get("data_profile")
            preprocessing = preprocessing_for(profile)
            construction = {
                "recipe": deepcopy(RECIPE),
                "prediction_unit": "clip",
                "selection_seed": settings.get("selection_seed", 1024),
                "data_profile": profile,
            }
            origin = {
                **reference_for("tall"),
                "selection_policy": "Remake source train/val only;no target-set checkpoint selection",
                "source_pool": "Declared all-frame manifest with hash-bound files-first metadata;not author-corpus discovery",
                "native_process_RNG_reproduced": False,
                "exact_optimizer_rng_resume_supported": False,
                "full_experiment_reproduction": False,
                "GPU_AMP_execution_verified": False,
                "native_constructor_repairs": "Undefined null-pretrained logging name andtwo CUDA zero allocations ported explicitly;math retained",
                "native_entrypoint": "dataset_type=tall falls through to generic base dataset;specialized cutout body requires explicit selector/collator repair",
            }
            weights = None
            if (
                settings.get("init_weights") is not None
                or settings.get("pretrained") is not None
            ):
                raise ValueError(
                    "TALL pretraining translation is not supported;use a complete initial_checkpoint or explicit random initialization"
                )
            if settings.get("initial_checkpoint") is not None:
                weights, receipt = trained_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif settings.get("initialization") != "random":
                raise ValueError(
                    "TALL needs explicit initialization=random or a complete initial_checkpoint"
                )
        self.validate_metadata(construction, preprocessing, origin)
        if (
            state is not None
            and settings.get("data_profile", construction["data_profile"])
            != construction["data_profile"]
        ):
            raise ValueError("TALL requested data recipe differs from saved Task")
        if state is not None or weights is not None:
            model = restore_detector(weights)
        else:
            model = TALLDetector(
                load_backbone("tall_swin_base", "vision", model_type="tall_swin_base")
            )
            origin["initialization"] = {
                "operation": "Native random construction;not training",
                "implicit_download": False,
                "native_pretrained_value": None,
            }
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
            raise ValueError("Unsupported or incomplete TALL Task state")

    @staticmethod
    def validate_metadata(construction, preprocessing, origin):
        if (
            not isinstance(construction, dict)
            or set(construction)
            != {"recipe", "prediction_unit", "selection_seed", "data_profile"}
            or construction["recipe"] != RECIPE
            or construction["prediction_unit"] != "clip"
            or type(construction["selection_seed"]) is not int
            or not 0 <= construction["selection_seed"] < 2**32
            or preprocessing != preprocessing_for(construction["data_profile"])
            or not isinstance(origin, dict)
        ):
            raise ValueError(
                "TALL requires its complete native T4 recipe and explicit data roles"
            )

    def get_collate_fn(self, stage):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown TALL preprocessing stage")
        return TALLCollator(profile=self.construction["data_profile"])

    def prepare_dataset(self, dataset, *, stage):
        return TALLDataset(
            dataset,
            stage=stage,
            seed=self.construction["selection_seed"],
            profile=self.construction["data_profile"],
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._ready = False
        report = TALLTrainer(self.model, origin=self.origin).fit(
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
            raise ValueError("Fit or import complete TALL weights before prediction")
        if runtime.precision != "fp32" or clips.dtype != torch.float32:
            raise ValueError("The selected native TALL alignment profile requires fp32")
        self.model.to(runtime.device).eval()
        scores = self.model.forward_outputs(
            clips.to(runtime.device, non_blocking=True)
        )["prob"]
        if (
            scores.shape != (len(clips),)
            or not torch.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
        ):
            raise ValueError("TALL must return one finite fake probability per clip")
        return scores


def create_task(settings, *, state=None):
    return TALLTask(settings, state=state)
