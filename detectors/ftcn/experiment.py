"""FTCN Task -> Detector -> lazy Backbone composition with strict clip state."""

from copy import deepcopy

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from detectors.i3d.dataset import I3DCollator, I3DDataset
from detectors.i3d.experiment import PREPROCESSING
from .checkpoint import initialization_weights, trained_weights
from .detector import FTCNDetector
from .trainer import FTCNTrainer


RECIPE = {
    "method": "ftcn",
    "backbone": "native_temporal_only_I3D_ResNet50_stop5",
    "clip_shape": [16, 3, 224, 224],
    "conversion": "native_dir_order_fresh_Conv3d_post_BN_stride_pool_no_time_pool",
    "initialization_order": "full_I3D_then_pretrained_copy_then_conversion_then_s5_discard_then_head",
    "head": "AvgPool1x14x14_C1024_T16_CLS_one_layer_16heads_dim64_MLP2048_literal_einsum_sigmoid",
    "features": "pooled_Nx1024x16x1x1",
    "objective": "probability_BCE",
    "prediction_unit": "clip",
    "frame_pool": "all_declared_source_frames_then_native_first32_T16_selection",
    "native_batch_size": 8,
    "scheduler": None,
    "precision_scope": "fp32_alignment_profile_not_native_CUDA_AMP_reproduction",
}


def build_model(*, initial_state=None):
    return FTCNDetector(load_backbone(
        "ftcn_resnet50", "vision", model_type="ftcn_resnet50",
        initial_state=initial_state,
    ))


class FTCNTask(ImageClassificationTask):
    prediction_unit = "clip"

    def __init__(self, settings, *, state=None):
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Native FTCN requires default float32 tensors")
        weights = initial_state = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (
                state["construction"], state["preprocessing"],
                state["origin"], state["state_dict"],
            )
            if any(value.is_floating_point() and value.dtype != torch.float32
                   for value in weights.values()):
                raise ValueError("Saved FTCN floating weights must remain float32")
        else:
            if settings.get("initial_checkpoint") and settings.get("init_weights"):
                raise ValueError("Select FTCN trained import or I3D initialization, not both")
            construction = {
                "recipe": deepcopy(RECIPE), "prediction_unit": "clip",
                "selection_seed": settings.get("selection_seed", 1024),
            }
            preprocessing = deepcopy(PREPROCESSING)
            origin = {
                **reference_for("ftcn"),
                "selection_policy": "Explicit source train/val; no target-set checkpoint selection",
                "source_pool": "Declared all-frame manifest and hash-bound files-first metadata",
                "entrypoint_repair": "Explicit FTCN factory; native ftcn.yaml incorrectly selects videomae",
                "unused_native_yaml_loss": "cross_entropy is ignored; native FTCN uses probability BCE",
                "native_process_RNG_reproduced": False,
                "exact_optimizer_rng_resume_supported": False,
                "native_generic_Trainer_use_amp_default": True,
                "GPU_AMP_execution_verified": False,
                "full_experiment_reproduction": False,
            }
            if settings.get("initial_checkpoint"):
                weights, receipt = trained_weights(settings["initial_checkpoint"])
                origin.update(receipt)
            elif settings.get("init_weights"):
                initial_state, receipt = initialization_weights(settings["init_weights"])
                origin["initialization"] = receipt
            elif settings.get("initialization") == "random":
                origin["initialization"] = "explicit_constructed_random"
            else:
                raise ValueError("FTCN needs init_weights or explicit initialization: random")
        if (
            set(construction) != {"recipe", "prediction_unit", "selection_seed"}
            or construction["recipe"] != RECIPE
            or construction["prediction_unit"] != "clip"
            or preprocessing != PREPROCESSING
            or type(construction["selection_seed"]) is not int
            or not 0 <= construction["selection_seed"] < 2**32
        ):
            raise ValueError("FTCN needs the complete native clip recipe and selection seed")
        if weights is not None:
            with torch.device("meta"):
                model = build_model()
            load_strict(model, weights, assign=True)
        else:
            model = build_model(initial_state=initial_state)
        super().__init__(model, construction=construction, preprocessing=preprocessing,
                         origin=origin, ready=weights is not None)

    def get_collate_fn(self, stage):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown FTCN preprocessing stage")
        return I3DCollator()

    def prepare_dataset(self, dataset, *, stage):
        return I3DDataset(dataset, stage=stage, seed=self.construction["selection_seed"])

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._ready = False
        report = FTCNTrainer(self.model, origin=self.origin).fit(
            loader, validation_loader=validation_loader, runtime=runtime,
            settings=settings, output_dir=output_dir,
        )
        self._ready = True
        return report

    @torch.inference_mode()
    def predict(self, clips, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import trained FTCN weights before prediction")
        if runtime.precision != "fp32":
            raise ValueError("The selected FTCN alignment profile requires fp32")
        self.model.to(runtime.device).eval()
        scores = self.model(clips.to(runtime.device, non_blocking=True)).view(-1)
        if (scores.shape != (len(clips),) or not torch.isfinite(scores).all()
                or not ((scores >= 0) & (scores <= 1)).all()):
            raise ValueError("FTCN must return one finite probability per clip")
        return scores


def create_task(settings, *, state=None):
    return FTCNTask(settings, state=state)
