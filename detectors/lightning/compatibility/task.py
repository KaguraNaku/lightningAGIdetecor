"""Lightning's experiment adapter; the shared engine knows none of these details."""

import json
import logging
import math
from pathlib import Path

import torch
from transformers import (
    AutoImageProcessor,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from backbones.load_backbone import load_backbone
from data.loader import ImageCollator
from engine.evaluate import coverage

from .calibration import calibrate, load_detector
from .encoder import CLIPVisionEncoder

logger = logging.getLogger(__name__)


def estimate_fit_bytes(samples, layers, channels, width, topk):
    """Estimate explicit buffers plus 25% workspace allowance, not a hard RSS cap.

    Includes raw features, projections/casts, per-layer temporaries and ridge
    systems. Excludes backbone weights, decoder caches and allocator overhead.
    """
    projected_dim = layers * width
    selected_dim = topk * width
    raw = samples * layers * channels * 4
    projected = samples * projected_dim * 20
    layer_work = samples * channels * 16 + channels * channels * 16
    systems = (projected_dim**2 + selected_dim**2) * 8 * 6
    return math.ceil((raw + projected + layer_work + systems + samples * 16) * 1.25)


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

    @torch.no_grad()
    def fit(self, loader, *, validation_loader=None, runtime, settings, output_dir):
        if loader.dataset.header["split"] != "train":
            raise ValueError("Lightning calibration requires a train manifest")
        if {record.label for record in loader.dataset.records} != {0, 1}:
            raise ValueError("Calibration requires both real and fake samples")
        cfg = self.config
        layers, channels = cfg["layers_count"], cfg["input_dim"]
        width, topk = cfg["output_dim"], cfg["topk"]
        if type(width) is not int or not 1 <= width <= channels:
            raise ValueError("output_dim must be between 1 and backbone hidden_size")
        if type(topk) is not int or not 1 <= topk <= layers:
            raise ValueError("topk must be between 1 and backbone layer count")
        budget = settings.get("feature_memory_budget_gib", 8)
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or budget <= 0
        ):
            raise ValueError("feature_memory_budget_gib must be finite and positive")
        count = len(loader.dataset)
        estimate = estimate_fit_bytes(count, layers, channels, width, topk)
        logger.info(
            "Estimated feature/fitting buffers %.2f GiB; budget %.2f GiB",
            estimate / 2**30,
            budget,
        )
        if estimate > budget * 2**30:
            raise MemoryError(
                f"Estimated feature/fitting buffers {estimate / 2**30:.2f} GiB exceed "
                f"feature_memory_budget_gib={budget}; use a reviewed smaller manifest "
                "or explicitly increase the budget"
            )
        features = torch.empty((count, layers, channels), dtype=torch.float32)
        labels = torch.empty(count, dtype=torch.long)
        seen = set()
        position = 0
        output_dir = Path(output_dir)
        self.encoder.to(runtime.device).eval()
        with (output_dir / "samples.jsonl").open("w") as stream:
            for batch in loader:
                with runtime.autocast():
                    batch_features = self.encoder(
                        batch["pixel_values"].to(runtime.device, non_blocking=True)
                    )
                batch_features = batch_features.detach().float().cpu()
                size = len(batch["labels"])
                if (
                    batch_features.shape != (size, layers, channels)
                    or not torch.isfinite(batch_features).all()
                ):
                    raise ValueError("Encoder returned invalid calibration features")
                if len(batch["metadata"]) != size or position + size > count:
                    raise ValueError("Calibration batch size or metadata mismatch")
                features[position : position + size].copy_(batch_features)
                labels[position : position + size].copy_(batch["labels"])
                for metadata, label in zip(
                    batch["metadata"], batch["labels"].tolist(), strict=True
                ):
                    sample_id = metadata["id"]
                    if sample_id in seen:
                        raise ValueError(f"Duplicate calibration sample: {sample_id}")
                    seen.add(sample_id)
                    stream.write(json.dumps({**metadata, "label": label}) + "\n")
                position += size
        report = coverage(loader.dataset, seen, output_dir)
        if not position:
            raise ValueError("No images successfully decoded for calibration")
        self.encoder.cpu()
        self.model = calibrate(
            self.encoder, features[:position], labels[:position], cfg
        )
        self._inference_device = None
        report.update(
            estimated_fit_bytes=estimate,
            selected_layers=self.model.mapping.selected_layers.tolist(),
        )
        logger.info(
            "Fitted %d samples; selected layers %s", position, report["selected_layers"]
        )
        return report

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
