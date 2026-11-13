"""Explicit author checkpoints or fresh CLIP initialization, no network fallback."""
from importlib.resources import files
import json
import torch
from backbones.load_backbone import load_backbone
from data.manifest import sha256
from detectors._shared.checkpoint import load_strict
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .data import ForAdaCollator, ForAdaDataset
from .detector import ForAdaDetector
from .trainer import ForAdaTrainer

DEFAULT_CLIP = dict(embed_dim=768, image_resolution=224, vision_layers=24,
                    vision_width=1024, vision_patch_size=14, context_length=77,
                    vocab_size=49408, transformer_width=768,
                    transformer_heads=12, transformer_layers=12)

def read_checkpoint(path):
    try:
        model = torch.jit.load(str(path), map_location="cpu")
        weights = model.state_dict()
    except (RuntimeError, ValueError):
        weights = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
    if not isinstance(weights, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()):
        raise ValueError("ForAda requires a plain tensor state dictionary or state_dict envelope")
    return weights, {"path": str(path), "sha256": sha256(path)}

class ForAdaTask(ImageClassificationTask):
    def __init__(self, settings, *, state=None):
        weights = None
        if state is not None:
            self.validate_state(state)
            construction, preprocessing, origin, weights = (state[k] for k in ("construction", "preprocessing", "origin", "state_dict"))
        else:
            if settings.get("augmentation", False):
                raise ValueError("Only the executable author augmentation=False profile is supported")
            construction = {"clip_config": settings.get("clip_config", DEFAULT_CLIP), "queries": 128,
                            "profile": "author-native-fp32-outofplace-v1"}
            preprocessing = preprocessing_config(256, (0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711), augmentation=False)
            origin = json.loads(files("detectors.forada").joinpath("reference.json").read_text())
            origin.update(objective="10CE+200XrayMSE+20AdapterIntra+10ClipContrastive",
                          fusion_policy="author_per_forward_unregistered_random_Fusion;global_torch_rng",
                          native_repairs=["singleton_preserve_batch_axis", "out_of_place_clip_patch_residual"],
                          checkpoint_replay="parameters_plus_matching_external_torch_rng;inference_is_native_stochastic")
            if settings.get("initial_checkpoint"):
                weights, origin["detector_checkpoint"] = read_checkpoint(settings["initial_checkpoint"])
            elif not settings.get("base_clip_checkpoint") and settings.get("initialization") != "random":
                raise ValueError("ForAda requires base_clip_checkpoint, initial_checkpoint, or explicit random fixture")
        if construction["profile"] != "author-native-fp32-outofplace-v1" or construction["queries"] != 128:
            raise ValueError("Unsupported ForAda construction profile")
        config = construction["clip_config"]
        if config["embed_dim"] != 768 or config["image_resolution"] != 224 or config["vision_patch_size"] != 14 or config["vision_layers"] < 15:
            raise ValueError("ForAda requires CLIP 224/14 geometry, >=15 visual layers and 768-dimensional projection")
        def build(device="cpu"):
            backbone = load_backbone("forada_clip", "full", model_type="forada_clip", **config)
            return ForAdaDetector(backbone, queries=construction["queries"], device=device)
        if weights is not None:
            with torch.device("meta"):
                model = build("meta")
            load_strict(model, weights, assign=True)
        else:
            model = build()
            if settings.get("base_clip_checkpoint"):
                base, identity = read_checkpoint(settings["base_clip_checkpoint"])
                for scalar in ("input_resolution", "context_length", "vocab_size"):
                    base.pop(scalar, None)
                model.clip_model.load_state_dict(base, strict=True)
                origin["base_clip_checkpoint"] = identity
            else:
                origin["initialization"] = "explicit_random_constructed_fixture"
        super().__init__(model, construction=construction, preprocessing=preprocessing,
                         origin=origin, ready=weights is not None)

    def get_collate_fn(self, stage):
        if stage not in ("train", "validation", "evaluation"):
            raise ValueError("Unknown ForAda stage")
        return ForAdaCollator(self.preprocessing, training=stage == "train")

    def prepare_dataset(self, dataset, *, stage):
        return ForAdaDataset(dataset) if stage == "train" else dataset

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        report = ForAdaTrainer(self.model, origin=self.origin).fit(
            loader, validation_loader=validation_loader, runtime=runtime,
            settings={"epochs": 60, **settings}, output_dir=output_dir)
        self._ready = True
        return report

def create_task(settings, *, state=None):
    return ForAdaTask(settings, state=state)
