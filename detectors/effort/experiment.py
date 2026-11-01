"""Effort adapter for the shared experiment pipeline."""

import torch
from transformers import AutoConfig, CLIPVisionConfig, CLIPVisionModel

from backbones.load_backbone import load_backbone
from data.loader import ImageCollator

from .detector import EffortDetector
from .transforms import make_transform, preprocessing_config

AUTHOR_COMMIT = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
AUTHOR_SHA256 = {
    "effort_detector.py": "366b1cde008f537e4b9c8c8e4c65ee20b430c4bca1ccee1b1b86c20a9831fac9",
    "abstract_dataset.py": "3682e97ffd1c83afdc261a5e8d35ca62d0fee61c71c317afc8ee281b6ff7f4d5",
    "effort.yaml": "1fd1398cf245b3a5c13cb130d7c6e209057ae6b56b561eca5f3c032283c5527b",
}
BACKBONE_ID = "openai/clip-vit-large-patch14"
BACKBONE_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


class EffortTask:
    def __init__(self, settings, *, state=None):
        self.settings = dict(settings)
        self.origin = {
            "author_commit": AUTHOR_COMMIT, "author_sha256": dict(AUTHOR_SHA256),
            "objective": "cross_entropy",
        }
        self.attention = settings.get("attention_implementation", "sdpa")
        if self.attention not in ("sdpa", "eager"):
            raise ValueError("Effort attention_implementation must be sdpa or eager")
        legacy = settings.get("initial_checkpoint") if state is None else None
        weights = None
        if state is not None:
            if state.get("version") != 1:
                raise ValueError("Unsupported Effort task checkpoint version")
            config = CLIPVisionConfig.from_dict(state["vision_config"])
            self.attention = state["attention_implementation"]
            rank = state["retained_rank"]
            self.preprocessing = state["preprocessing"]
            self.origin = state["origin"]
            weights = state["state_dict"]
        else:
            model_id = settings.get("backbone_id", BACKBONE_ID)
            options = {"local_files_only": settings.get("local_files_only", False)}
            revision = settings.get("revision")
            if revision is None and model_id == BACKBONE_ID:
                revision = BACKBONE_REVISION
            if revision is not None:
                options["revision"] = revision
            if legacy:
                # Import requires only geometry; never initialize pretrained weights/SVD.
                config = AutoConfig.from_pretrained(model_id, **options)
                config = getattr(config, "vision_config", config)
                from .checkpoint import read_legacy

                weights, receipt = read_legacy(legacy)
                self.origin.update(receipt)
            else:
                vision = load_backbone(
                    model_id, "vision", attn_implementation=self.attention, **options
                )
                config = vision.config
            rank = settings.get("retained_rank", config.hidden_size - 1)
            self.preprocessing = preprocessing_config(config.image_size)
        self.vision_config = config.to_dict()
        if weights is not None:
            config._attn_implementation = self.attention
            with torch.device("meta"):
                self.model = EffortDetector(
                    CLIPVisionModel(config), retained_rank=rank, initialize=False
                )
            self.model.load_state_dict(weights, strict=True, assign=True)
            # HF keeps position_ids as a nonpersistent buffer, absent from weights.
            embeddings = self.model.backbone.embeddings
            embeddings.position_ids = torch.arange(
                embeddings.position_embedding.num_embeddings, device="cpu"
            ).expand((1, -1))
        else:
            self.model = EffortDetector(vision, retained_rank=rank)
        self.collate_fn = ImageCollator(transform=make_transform(self.preprocessing))
        self._ready = weights is not None
        self._device = None

    def get_collate_fn(self, stage):
        if stage not in ("train", "validation", "evaluation"):
            raise ValueError(f"Unknown Effort preprocessing stage: {stage}")
        if stage == "train":
            return ImageCollator(transform=make_transform(self.preprocessing, training=True))
        return self.collate_fn

    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        from .trainer import EffortTrainer

        self._device = None
        report = EffortTrainer(self.model, origin=self.origin).fit(
            loader, validation_loader=validation_loader,
            runtime=runtime, settings=settings, output_dir=output_dir,
        )
        self._ready = True
        return report

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if not self._ready:
            raise ValueError("Fit or import Effort weights before prediction")
        if self._device != runtime.device:
            self.model.to(runtime.device)
            self._device = runtime.device
        self.model.eval()
        with runtime.autocast():
            return self.model(images.to(runtime.device, non_blocking=True)).softmax(dim=1)[:, 1]

    def state_dict(self):
        if not self._ready:
            raise ValueError("Cannot save an unfitted Effort model")
        return {
            "version": 1,
            "vision_config": self.vision_config,
            "attention_implementation": self.attention,
            "retained_rank": self.model.retained_rank,
            "preprocessing": self.preprocessing,
            "origin": self.origin,
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }


def create_task(settings, *, state=None):
    return EffortTask(settings, state=state)
