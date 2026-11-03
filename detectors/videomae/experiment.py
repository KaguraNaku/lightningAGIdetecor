"""VideoMAETask -> Detector -> lazily selected independent base encoder."""

from copy import deepcopy
from importlib.resources import files
import json

import torch

from backbones.load_backbone import load_backbone
from detectors._shared.checkpoint import load_strict
from detectors._shared.provenance import reference_for
from detectors._shared.task import ImageClassificationTask
from .checkpoint import initialization_weights, trained_weights, restore_detector, restore_positions
from .dataset import VideoMAEDataset, VideoMAECollator
from .detector import VideoMAEDetector
from .trainer import VideoMAETrainer

SPEC = json.loads(files("configs").joinpath("migrations/videomae_base.json").read_text())
RECIPE = {
    "method": "videomae", "model_id": SPEC["model_id"], "revision": SPEC["revision"],
    "config": SPEC["config"], "attention_profile": SPEC["selected_attention_profile"],
    "clip_shape": [16, 3, 224, 224], "prediction_unit": "clip",
    "head": "internal_token_LN1e-12_then_all_token_mean_then_LN1e-5_Linear768to2",
    "objective": "two_logit_cross_entropy", "native_batch_size": 8,
    "scheduler": None, "precision": "fp32",
}
PREPROCESSING = {
    "resolution": 224, "clip_size": 16, "channels": "RGB",
    "resize": "OpenCV_INTER_CUBIC", "mean": [0.5]*3, "std": [0.5]*3,
    "frame_pool": "declared_complete_source_frames_native_first32_T16_selection",
    "with_mask": False, "with_landmark": True,
    "landmarks": "filesystem_native_literal_path_NPY_unscaled_or_absent81xy_float64_zeros;hash_bound_at_dataset_construction",
    "landmark_output": "original_untransformed_NPY_tensor_like_native_getitem",
    "augmentation": "native_Albu1.3.1_keypoint_conditioned_shared_clip_seed",
    "validation": "native_augmented_validation;unaugmented_evaluation",
    "worker_RNG": "restore_Python_NumPy_instead_of_native_entropy_reseed",
    "decode_failure": "fatal_one_attempt_no_label_or_clip_substitution",
}


def build_model(*, encoder_state=None):
    if encoder_state is None:
        backbone = load_backbone("videomae_base_native", "vision", model_type="videomae_base_native")
    else:
        # Complete pretrained copy should not consume random constructor draws
        # before initializing the external head. Meta + strict assign preserves
        # that from_pretrained behavior; its numeric validation remains separate.
        with torch.device("meta"):
            backbone = load_backbone("videomae_base_native", "vision", model_type="videomae_base_native")
        load_strict(backbone, encoder_state, assign=True)
        restore_positions(backbone)
        backbone.eval()  # Native HF from_pretrained returns an eval backbone.
    return VideoMAEDetector(backbone)


class VideoMAETask(ImageClassificationTask):
    prediction_unit = "clip"

    def __init__(self, settings, *, state=None):
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Native VideoMAE profile requires default float32")
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin = state["construction"], state["preprocessing"], state["origin"]
        else:
            construction = {"recipe": deepcopy(RECIPE), "prediction_unit": "clip",
                            "selection_seed": settings.get("selection_seed", 1024)}
            preprocessing = deepcopy(PREPROCESSING)
            origin = {**reference_for("videomae"),
                      "selection_policy": "Source-only train/val;no target-set checkpoint selection",
                      "published_weights_used_by_default": False,
                      "native_yaml_pretrained_path": "ignored stale Xception path;explicit VideoMAE init_weights required",
                      "native_global_worker_RNG_reproduced": False,
                      "native_auto_attention_backend_reproduced": False,
                      "exact_optimizer_rng_resume_supported": False,
                      "GPU_AMP_execution_verified": False, "full_experiment_reproduction": False}
        if (set(construction) != {"recipe", "prediction_unit", "selection_seed"}
                or construction["recipe"] != RECIPE or construction["prediction_unit"] != "clip"
                or preprocessing != PREPROCESSING or type(construction["selection_seed"]) is not int
                or not 0 <= construction["selection_seed"] < 2**32):
            raise ValueError("VideoMAE requires its complete native clip recipe,preprocessing and selection seed")
        weights = None
        if state is not None:
            weights = state["state_dict"]
        elif settings.get("initial_checkpoint") and settings.get("init_weights"):
            raise ValueError("Select VideoMAE trained import or pretrained initialization,not both")
        elif settings.get("initial_checkpoint"):
            weights, receipt = trained_weights(settings["initial_checkpoint"])
            origin.update(receipt)
        elif settings.get("init_weights"):
            encoder, receipt = initialization_weights(settings["init_weights"])
            model = build_model(encoder_state=encoder)
            origin["initialization"] = receipt
        elif settings.get("initialization") == "random":
            model = build_model()
            origin["initialization"] = "explicit_constructed_random"
        else:
            raise ValueError("VideoMAE needs init_weights or explicit initialization: random")
        if weights is not None:
            with torch.device("meta"):
                model = build_model()
            restore_detector(model, weights)
        super().__init__(model, construction=construction, preprocessing=preprocessing,
                         origin=origin, ready=weights is not None)

    def get_collate_fn(self, stage):
        if stage not in {"train", "validation", "evaluation"}:
            raise ValueError("Unknown VideoMAE preprocessing stage")
        return VideoMAECollator()

    def prepare_dataset(self, dataset, *, stage):
        return VideoMAEDataset(dataset, stage=stage, seed=self.construction["selection_seed"])

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        self._ready = False
        report = VideoMAETrainer(self.model, origin=self.origin).fit(
            loader, validation_loader=validation_loader, runtime=runtime,
            settings=settings, output_dir=output_dir)
        self._ready = True
        return report

    @torch.inference_mode()
    def predict(self, clips, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import trained VideoMAE weights before prediction")
        if runtime.precision != "fp32":
            raise ValueError("Selected VideoMAE profile requires fp32")
        self.model.to(runtime.device).eval()
        logits = self.model(clips.to(runtime.device, non_blocking=True))
        if logits.shape != (len(clips), 2) or not torch.isfinite(logits).all():
            raise ValueError("VideoMAE must return two finite logits per clip")
        return logits.softmax(1)[:, 1]


def create_task(settings, *, state=None):
    return VideoMAETask(settings, state=state)
