"""Read historical checkpoints; fitting uses only the formal uniform-ridge Task."""

import torch
from transformers import (
    AutoImageProcessor,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from backbones.load_backbone import load_backbone
from data.loader import ImageCollator

from .calibration import load_detector
from .encoder import CLIPVisionEncoder

class LightningTask:
    def __init__(self, settings, *, state=None):
        self.settings = settings
        self.model = None
        self._inference_device = None
        if state is None:
            model_id = settings["backbone_id"]
            vision = load_backbone(model_id, "vision")
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.encoder = CLIPVisionEncoder(vision, model_id=model_id)
            self.config = {
                "layers_count": vision.config.num_hidden_layers,
                "input_dim": vision.config.hidden_size,
                "output_dim": settings.get("output_dim", 64),
                "eps": settings.get("eps", 1e-6),
                "topk": settings.get("topk", 8),
                "search_ridge_lambda": settings.get("search_ridge_lambda", 1.0),
                "ridge_lambda": settings.get("ridge_lambda", 0.15),
            }
        else:
            if state.get("version") != 1:
                raise ValueError("Unsupported Lightning task checkpoint version")
            vision = CLIPVisionModel(CLIPVisionConfig.from_dict(state["vision_config"]))
            self.encoder = CLIPVisionEncoder(vision, model_id=state["encoder_id"])
            self.processor = CLIPImageProcessor.from_dict(state["processor"])
            self.config = state["config"]
            self.model = load_detector(self.encoder, self.config, state["state_dict"])
        if state is None and settings.get("selected_layers") is not None:
            self.config["selected_layers"] = settings["selected_layers"]
        self.collate_fn = ImageCollator(self.processor)

    def fit(self, *args, **kwargs):
        raise ValueError("Historical Lightning is inference-only; use the formal uniform-ridge Task")

    def get_collate_fn(self, stage):
        return self.collate_fn

    @torch.inference_mode()
    def predict(self, images, *, runtime):
        if self.model is None:
            raise ValueError("Fit or restore the model before prediction")
        if self._inference_device != runtime.device:
            self.model.to(runtime.device).eval()
            self._inference_device = runtime.device
        with runtime.autocast():
            features = self.model.backbone(images.to(runtime.device, non_blocking=True))
        # Autocast applies only to the encoder, never to mapping/ridge arithmetic.
        return self.model.classifier(self.model.mapping(features.float()))

    def state_dict(self):
        if self.model is None:
            raise ValueError("Cannot save an unfitted model")
        state = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in self.model.state_dict().items()
        }
        return {
            "version": 1,
            "config": self.config,
            "encoder_id": self.encoder.model_id,
            "vision_config": self.encoder.model.config.to_dict(),
            "processor": self.processor.to_dict(),
            "state_dict": state,
        }


def create_task(settings, *, state=None):
    return LightningTask(settings, state=state)
