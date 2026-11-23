"""Effort adapter for the shared experiment pipeline."""

import torch
from transformers import AutoConfig, CLIPVisionConfig, CLIPVisionModel

from backbones.load_backbone import load_backbone
from data.loader import ImageCollator
from detectors._shared.checkpoint import restore_model
from detectors._shared.task import ImageClassificationTask
from models.clip import initialize_clip_position_ids

from .detector import EffortDetector
from .trainer import EffortTrainer
from .transforms import make_transform, preprocessing_config

AUTHOR_COMMIT = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
AUTHOR_SHA256 = {
    "effort_detector.py": "366b1cde008f537e4b9c8c8e4c65ee20b430c4bca1ccee1b1b86c20a9831fac9",
    "abstract_dataset.py": "3682e97ffd1c83afdc261a5e8d35ca62d0fee61c71c317afc8ee281b6ff7f4d5",
    "effort.yaml": "1fd1398cf245b3a5c13cb130d7c6e209057ae6b56b561eca5f3c032283c5527b",
}
BACKBONE_ID = "openai/clip-vit-large-patch14"
BACKBONE_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


class EffortTask(ImageClassificationTask):
    trainer_class = EffortTrainer

    def __init__(self, settings, *, state=None):
        self.attention = settings.get("attention_implementation", "sdpa")
        if self.attention not in ("sdpa", "eager"):
            raise ValueError("Effort attention_implementation must be sdpa or eager")

        if state is None:
            model, config, origin, imported = self._new_model(settings)
            preprocessing = preprocessing_config(config.image_size)
        else:
            self.validate_state(state)
            self.attention = state["attention_implementation"]
            config = CLIPVisionConfig.from_dict(state["vision_config"])
            model = self._restore_model(
                config, state["retained_rank"], state["state_dict"]
            )
            preprocessing = state["preprocessing"]
            origin = state["origin"]
            imported = True

        super().__init__(
            model, preprocessing=preprocessing, origin=origin, ready=imported
        )
        self._device = None

    def _new_model(self, settings):
        origin = {
            "author_commit": AUTHOR_COMMIT,
            "author_sha256": dict(AUTHOR_SHA256),
            "objective": "cross_entropy",
        }
        model_id = settings.get("backbone_id", BACKBONE_ID)
        options = {"local_files_only": settings.get("local_files_only", False)}
        revision = settings.get("revision")
        if revision is None and model_id == BACKBONE_ID:
            revision = BACKBONE_REVISION
        if revision is not None:
            options["revision"] = revision

        if checkpoint := settings.get("initial_checkpoint"):
            from .checkpoint import read_legacy

            # Imported weights already contain the learned decomposition.
            config = AutoConfig.from_pretrained(model_id, **options)
            config = getattr(config, "vision_config", config)
            weights, receipt = read_legacy(checkpoint)
            origin.update(receipt)
            rank = settings.get("retained_rank", config.hidden_size - 1)
            return self._restore_model(config, rank, weights), config, origin, True

        vision = load_backbone(
            model_id, "vision", attn_implementation=self.attention, **options
        )
        config = vision.config
        self.vision_config = config.to_dict()
        rank = settings.get("retained_rank", config.hidden_size - 1)
        return EffortDetector(vision, retained_rank=rank), config, origin, False

    def _restore_model(self, config, rank, weights):
        self.vision_config = config.to_dict()
        config._attn_implementation = self.attention

        def build():
            return EffortDetector(
                CLIPVisionModel(config), retained_rank=rank, initialize=False
            )

        return restore_model(
            build, weights, initialize_buffers=initialize_clip_position_ids
        )

    def build_collator(self, *, training):
        return ImageCollator(
            transform=make_transform(self.preprocessing, training=training)
        )

    def fit(self, loader, **kwargs):
        self._device = None
        return super().fit(loader, **kwargs)

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import Effort weights before prediction")
        if self._device != runtime.device:
            self.model.to(runtime.device)
            self._device = runtime.device
        self.model.eval()
        with runtime.autocast():
            return self.model(images.to(runtime.device, non_blocking=True)).softmax(
                dim=1
            )[:, 1]

    def checkpoint_metadata(self):
        return {
            "vision_config": self.vision_config,
            "attention_implementation": self.attention,
            "retained_rank": self.model.retained_rank,
            "preprocessing": self.preprocessing,
            "origin": self.origin,
        }
