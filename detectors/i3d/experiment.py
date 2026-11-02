"""I3D Task composition; native numeric/workflow validation remains separate."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict, read_weights
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from .checkpoint import initialize_backbone
from .dataset import I3DDataset, I3DCollator
from .detector import I3DDetector
from .trainer import I3DTrainer


RECIPE = {
    "backbone": "native_single_path_I3D_ResNet50",
    "clip_shape": [16, 3, 224, 224],
    "frame_pool": "all_declared_source_frames_then_native_first32_T16_selection",
    "head": "fixed_avgpool8x7x7_dropout0.5_linear2048to1_sigmoid_train_and_eval",
    "objective": "probability_BCE",
    "prediction_unit": "clip",
    "native_batch_size": 8,
    "scheduler": None,
    "precision_scope": "fp32_alignment_profile_not_native_CUDA_AMP_reproduction",
}
PREPROCESSING = {
    "resolution": 224,
    "clip_size": 16,
    "channels": "RGB",
    "resize": "OpenCV_INTER_CUBIC",
    "mean": [0.5] * 3,
    "std": [0.5] * 3,
    "augmentation": "native_Albu1.3.1_shared_clip_seed",
    "worker_RNG": "restore_Python_NumPy_state_instead_of_native_entropy_reset",
    "validation": "evaluation_profile_not_native_optional_augmented_val",
    "decode_failure": "fatal_one_attempt_no_clip_video_or_label_substitution",
    "with_mask": False,
    "with_landmark": False,
}


def build_model():
    return I3DDetector(
        load_backbone("i3d_resnet50", "vision", model_type="i3d_resnet50")
    )


def trained_weights(path):
    weights, origin = read_weights(path, encoding="latin1")
    if all(name.startswith("resnet.") for name in weights):
        layout = "native_detector"
    elif (
        not any(name.startswith("resnet.") for name in weights)
        and "head.projection.weight" in weights
    ):
        weights = {f"resnet.{name}": value for name, value in weights.items()}
        layout = "native_backbone"
    else:
        raise ValueError(
            "I3D initial_checkpoint needs an unambiguous detector/backbone layout"
        )
    head = weights.get("resnet.head.projection.weight")
    if head is None or head.shape != (1, 2048):
        raise ValueError(
            "I3D trained checkpoint needs a1-row head; use init_weights for Kinetics initialization"
        )
    # The native constructor creates float32 modules before copying checkpoint
    # weights. Preserve that import dtype policy without executing legacy code.
    weights = {
        name: value.float() if value.is_floating_point() else value
        for name, value in weights.items()
    }
    return weights, {
        **origin,
        "layout": layout,
        "floating_import_dtype": "float32_as_native_constructor",
        "native_pickle_encoding": "latin1",
    }


class I3DTask(ImageClassificationTask):
    prediction_unit = "clip"

    def __init__(self, settings, *, state=None):
        if torch.get_default_dtype() != torch.float32:
            raise ValueError(
                "The selected native I3D recipe requires default float32 tensors"
            )
        weights = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"],
                state["preprocessing"],
                state["origin"],
                state["state_dict"],
            )
            if any(
                value.is_floating_point() and value.dtype != torch.float32
                for value in weights.values()
            ):
                raise ValueError("Saved native I3D task weights must remain float32")
        else:
            construction = {
                "recipe": deepcopy(RECIPE),
                "prediction_unit": "clip",
                "selection_seed": settings.get("selection_seed", 1024),
            }
            preprocessing = deepcopy(PREPROCESSING)
            origin = {
                **reference_for("i3d"),
                "selection_policy": "Remake explicit source train/val protocol; no target-set checkpoint selection",
                "source_pool": "Declared all-frame manifest plus hash-bound files-first metadata; not fresh corpus discovery",
                "native_process_RNG_reproduced": False,
                "exact_optimizer_rng_resume_supported": False,
                "full_experiment_reproduction": False,
                "native_generic_Trainer_use_amp_default": True,
                "GPU_AMP_execution_verified": False,
                "unused_native_yaml_loss": "cross_entropy is ignored by the original I3D constructor; actual objective is probability nn.BCELoss",
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = trained_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif (
                not settings.get("init_weights")
                and settings.get("initialization") != "random"
            ):
                raise ValueError(
                    "I3D requires init_weights or explicit initialization: random"
                )
        if (
            set(construction) != {"recipe", "prediction_unit", "selection_seed"}
            or construction["recipe"] != RECIPE
            or construction["prediction_unit"] != "clip"
            or preprocessing != PREPROCESSING
            or type(construction["selection_seed"]) is not int
            or not 0 <= construction["selection_seed"] < 2**32
        ):
            raise ValueError(
                "I3D requires its complete native clip recipe and explicit selection seed"
            )
        if weights is not None:
            with torch.device("meta"):
                model = build_model()
            load_strict(model, weights, assign=True)
        else:
            model = build_model()
            origin["initialization"] = (
                initialize_backbone(model.resnet, settings["init_weights"])
                if settings.get("init_weights")
                else "explicit_constructed_random"
            )
        super().__init__(
            model,
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    def get_collate_fn(self, stage):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown I3D preprocessing stage")
        return I3DCollator()

    def prepare_dataset(self, dataset, *, stage):
        return I3DDataset(
            dataset, stage=stage, seed=self.construction["selection_seed"]
        )

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._ready = False
        report = I3DTrainer(self.model, origin=self.origin).fit(
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
            raise ValueError("Fit or import trained I3D weights before prediction")
        if runtime.precision != "fp32":
            raise ValueError("The selected I3D score-alignment recipe requires fp32")
        self.model.to(runtime.device).eval()
        scores = self.model(clips.to(runtime.device, non_blocking=True)).view(-1)
        if (
            scores.shape != (len(clips),)
            or not torch.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()
        ):
            raise ValueError("I3D must return one finite probability per clip")
        return scores


def create_task(settings, *, state=None):
    return I3DTask(settings, state=state)
