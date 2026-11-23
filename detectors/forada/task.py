"""Explicit author checkpoints or fresh CLIP initialization, no network fallback."""
from importlib.resources import files
from pathlib import Path
import json
import torch
from data.manifest import sha256
from detectors._shared.checkpoint import restore_model
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from .clip import load_forada_clip
from .data import FakeGuardSampler, ForAdaCollator, ForAdaDataset, WholeImageForAdaDataset
from .detector import ForAdaDetector
from .trainer import ForAdaTrainer

DEFAULT_CLIP = dict(embed_dim=768, image_resolution=224, vision_layers=24,
                    vision_width=1024, vision_patch_size=14, context_length=77,
                    vocab_size=49408, transformer_width=768,
                    transformer_heads=12, transformer_layers=12)

def read_checkpoint(path):
    path = Path(path).expanduser().resolve()
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
    trainer_class = ForAdaTrainer
    default_epochs = 60
    model_file_fields = ("base_clip_checkpoint",)
    import_ignored_settings = model_file_fields

    @staticmethod
    def configure_ufd_training(adaptation, *, warm_start):
        adaptation["ForAda_supervision"] = "whole-image masks for fully generated images"

    @staticmethod
    def inspect_training_input(model, fit, training, validation):
        if fit.get("ufd") is not None:
            summary = WholeImageForAdaDataset(training).selection_receipt()
        else:
            masks = ForAdaDataset(
                training, require_localized=model.get("training_profile") == "author_ffpp_8_32",
            )
            summary = masks.supervision_summary
        return {"mask_supervision": summary}

    def __init__(self, settings, *, state=None):
        base_checkpoint = None
        if state is None:
            construction, preprocessing, origin = self._training_config(settings)
            weights = None
            if checkpoint := settings.get("initial_checkpoint"):
                weights, origin["detector_checkpoint"] = read_checkpoint(checkpoint)
            else:
                base_checkpoint = settings.get("base_clip_checkpoint")
                if not base_checkpoint and settings.get("initialization") != "random":
                    raise ValueError(
                        "ForAda requires base_clip_checkpoint, initial_checkpoint, or initialization: random"
                    )
        else:
            construction, preprocessing, origin, weights = self.unpack_state(state)

        model = self._build_model(construction, weights, base_checkpoint, origin)
        super().__init__(
            model, construction=construction, preprocessing=preprocessing,
            origin=origin, ready=weights is not None,
        )
        self.author_source = (
            settings.get("training_profile") == "author_ffpp_8_32"
            or origin.get("source_supervision") == "localized_author_masks"
        )
        if self.author_source:
            self.origin["source_supervision"] = "localized_author_masks"

    @staticmethod
    def _training_config(settings):
        if settings.get("augmentation", False):
            raise ValueError("Only the executable author augmentation=False profile is supported")
        construction = {
            "clip_config": settings.get("clip_config", DEFAULT_CLIP),
            "queries": 128,
            "profile": "author-native-fp32-outofplace-v1",
        }
        preprocessing = preprocessing_config(
            256, (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711), augmentation=False,
        )
        origin = json.loads(files("detectors.forada").joinpath("reference.json").read_text())
        origin.update(
            objective="10CE+200XrayMSE+20AdapterIntra+10ClipContrastive",
            fusion_policy="author_per_forward_unregistered_random_Fusion;global_torch_rng",
            native_repairs=["singleton_preserve_batch_axis", "out_of_place_clip_patch_residual"],
            checkpoint_replay="parameters_plus_matching_external_torch_rng;inference_is_native_stochastic",
        )
        return construction, preprocessing, origin

    @staticmethod
    def _build_model(construction, weights, base_checkpoint, origin):
        if construction["profile"] != "author-native-fp32-outofplace-v1" or construction["queries"] != 128:
            raise ValueError("Unsupported ForAda construction profile")
        config = construction["clip_config"]
        if (
            config["embed_dim"] != 768 or config["image_resolution"] != 224
            or config["vision_patch_size"] != 14 or config["vision_layers"] < 15
        ):
            raise ValueError("ForAda requires CLIP 224/14 geometry, >=15 visual layers and 768-dimensional projection")

        def build(device="cpu"):
            backbone = load_forada_clip("forada_clip", **config)
            return ForAdaDetector(backbone, queries=construction["queries"], device=device)

        if weights is not None:
            return restore_model(lambda: build("meta"), weights)

        model = build()
        if base_checkpoint:
            base, identity = read_checkpoint(base_checkpoint)
            for scalar in ("input_resolution", "context_length", "vocab_size"):
                base.pop(scalar, None)
            model.clip_model.load_state_dict(base, strict=True)
            origin["base_clip_checkpoint"] = identity
        else:
            origin["initialization"] = "explicit_random_constructed_fixture"
        return model

    def build_collator(self, *, training):
        return ForAdaCollator(self.preprocessing, training=training)

    def prepare_dataset(self, dataset, *, stage):
        if stage != "train":
            return dataset
        if "UFD_adaptation" in self.origin:
            return WholeImageForAdaDataset(dataset)
        return ForAdaDataset(dataset, require_localized=self.author_source)

    def get_sampler(self, dataset, *, batch_size, shuffle, seed, stage):
        if stage == "train" and "UFD_adaptation" in self.origin:
            return FakeGuardSampler(dataset.records, batch_size, seed)
        return None
