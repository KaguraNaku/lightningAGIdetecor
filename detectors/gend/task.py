"""Pinned GenD CLIP-L/14 task, with explicit head semantics on raw conversion."""

from huggingface_hub import hf_hub_download

from detectors._shared.checkpoint import restore_model
from detectors._shared.task import ImageClassificationTask
from detectors._shared.transforms import preprocessing_config
from models.clip import initialize_clip_position_ids
from .checkpoint import read_gend_weights
from .detector import GenDDetector
from .encoder import GenDCLIPEncoder


MODEL_ID = "yermandy/GenD_CLIP_L_14"
REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"
DEFAULT_HEAD = "LinearNorm"
DEFAULT_VISION_CONFIG = {
    "image_size": 224,
    "patch_size": 14,
    "num_channels": 3,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "projection_dim": 768,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "initializer_factor": 1.0,
    "return_dict": True,
}


class GenDTask(ImageClassificationTask):
    default_epochs = 11

    def __init__(self, settings, *, state=None):
        backbone = None
        if state is None:
            construction, preprocessing, origin = self._training_config(settings)
            weights, backbone, weight_origin = self._initial_weights(settings, construction)
            origin.update(weight_origin)
        else:
            construction, preprocessing, origin, weights = self.unpack_state(state)
        self._configure_training(construction, preprocessing)
        super().__init__(
            self._build_model(construction, weights, backbone),
            construction=construction,
            preprocessing=preprocessing,
            origin=origin,
            ready=weights is not None,
        )

    @staticmethod
    def _training_config(settings):
        construction = {
            "vision_config": settings.get("vision_config", DEFAULT_VISION_CONFIG.copy()),
            "head": settings.get("head", DEFAULT_HEAD),
            "attention_implementation": settings.get("attention_implementation", "sdpa"),
            "freeze_policy": "feature_extractor_norm_or_ln_names_plus_complete_head",
            "forward_precision": "float32_autocast_disabled",
            "training_profile": settings.get("training_profile", "author_ln_l2_ua"),
        }
        preprocessing = preprocessing_config(
            construction["vision_config"]["image_size"],
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
            augmentation=settings.get("augmentation", False),
            data_aug=settings.get("data_aug"),
        )
        origin = {
            "method": "gend",
            "objective": "cross_entropy",
            "backbone_family": "CLIP",
        }
        if construction["training_profile"] == "author_ln_l2_ua":
            if construction["head"] != "LinearNorm":
                raise ValueError("Full GenD requires its L2-normalized linear head")
            if settings.get("data_aug") is not None or settings.get("augmentation") is False:
                raise ValueError("GenD author profile fixes its PIL augmentation policy")
            preprocessing.pop("data_aug", None)
            preprocessing["profile"] = "gend-author-pil-v1"
            preprocessing["augmentation"] = True
            origin.update(
                objective="CE+.5*uniformity+.1*label_alignment",
                author_revision="a31acda1f8c6017b4753a8b39bfb9214127ad3b3",
                protocol="author method under common train8/test32 source-only validation",
            )
        return construction, preprocessing, origin

    @staticmethod
    def _initial_weights(settings, construction):
        if checkpoint := settings.get("initial_checkpoint"):
            if "head" not in settings:
                raise ValueError("Raw GenD imports require explicit head: linear or LinearNorm")
            weights, receipt = read_gend_weights(checkpoint)
            return weights, None, receipt
        if settings.get("initialization") == "pretrained_backbone":
            from transformers import CLIPVisionModelWithProjection

            backbone_id = settings.get("backbone_id")
            if not backbone_id:
                raise ValueError("Fresh GenD training requires backbone_id")
            revision = settings.get("revision")
            backbone = CLIPVisionModelWithProjection.from_pretrained(
                backbone_id,
                revision=revision,
                local_files_only=True,
                attn_implementation=construction["attention_implementation"],
            )
            actual = backbone.config.to_dict()
            geometry = (
                "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "image_size", "patch_size", "projection_dim",
            )
            for key in geometry:
                if actual[key] != construction["vision_config"][key]:
                    raise ValueError("GenD base CLIP geometry differs: " + key)
            return None, backbone, {
                "initialization": "fresh_pretrained_backbone_random_head",
                "backbone_id": str(backbone_id),
                "revision": revision,
            }
        if settings.get("initialization") == "random":
            return None, None, {"initialization": "explicit_constructed_random"}

        if settings.get("vision_config") is not None or construction["head"] != DEFAULT_HEAD:
            raise ValueError(
                "Custom GenD geometry/head requires random initialization or explicit raw checkpoint"
            )
        path = hf_hub_download(
            MODEL_ID, "model.safetensors", revision=REVISION, local_files_only=True
        )
        weights, receipt = read_gend_weights(path)
        return weights, None, {
            **receipt,
            "pretrained_model": {"model_id": MODEL_ID, "revision": REVISION, "head": DEFAULT_HEAD},
        }

    def _configure_training(self, construction, preprocessing):
        if construction["attention_implementation"] not in ("eager", "sdpa"):
            raise ValueError("GenD attention implementation must be eager or sdpa")
        if (
            construction["freeze_policy"]
            != "feature_extractor_norm_or_ln_names_plus_complete_head"
            or construction["forward_precision"] != "float32_autocast_disabled"
        ):
            raise ValueError("GenD state must retain native freezing and precision policy")
        profile = construction.get("training_profile", "legacy_ce")
        if profile not in ("legacy_ce", "author_ln_l2_ua"):
            raise ValueError("Unknown GenD training profile")
        if profile == "author_ln_l2_ua":
            if construction["head"] != "LinearNorm" or preprocessing.get("profile") != "gend-author-pil-v1":
                raise ValueError("Full GenD state requires its normalized head and author preprocessing")
            from .trainer import GenDTrainer

            self.trainer_class = GenDTrainer
            self.default_epochs = 30

    @staticmethod
    def _build_model(construction, weights, backbone):
        def build():
            encoder = GenDCLIPEncoder(
                vision_config=construction["vision_config"],
                attention_implementation=construction["attention_implementation"],
            )
            return GenDDetector(encoder, head=construction["head"])

        if weights is not None:
            model = restore_model(build, weights, initialize_buffers=initialize_clip_position_ids)
            # Keep trainable tensors independent of the input state dictionary.
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.detach().clone()
            return model

        model = build()
        if backbone is not None:
            model.model.feature_extractor.load_state_dict(backbone.state_dict(), strict=True)
        return model

    def validate_training_parent(self, settings):
        expected = settings.get("training_profile", "author_ln_l2_ua")
        if self.construction.get("training_profile", "legacy_ce") != expected:
            raise ValueError("GenD parent training profile differs; a historical CE-only model cannot become full GenD silently")

    def build_collator(self, *, training):
        if self.preprocessing["profile"] == "gend-author-pil-v1":
            from data.loader import ImageCollator
            from .transforms import GenDTransform
            return ImageCollator(transform=GenDTransform(self.preprocessing, training=training))
        return super().build_collator(training=training)
