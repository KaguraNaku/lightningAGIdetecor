"""Native CE+BCE+MSE with synchronized RGB/IDCT/masks and common fit/selection."""

import json
from pathlib import Path

import torch

from detectors._shared.trainer import ImageClassificationTrainer
from engine.runtime import atomic_save
from .dataset import LRLDataset, LRLCollator, LRLDataError


class LRLScorer:
    def __init__(self, model):
        self.model = model

    def predict_batch(self, batch, *, runtime):
        with runtime.autocast():
            return self.model.forward_outputs(
                batch["pixel_values"].to(runtime.device, non_blocking=True),
                batch["idct"].to(runtime.device, non_blocking=True),
            )["prob"]


class LRLTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin)
        self.component_sums, self.component_count = {}, 0

    def make_scorer(self):
        return LRLScorer(self.model)

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        frequency, masks = batch.get("idct"), batch.get("masks")
        for value, name in ((frequency, "IDCT"), (masks, "mask")):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (len(labels), 1, 256, 256)
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"LRL requires aligned finite float32 {name}256 inputs"
                )
        outputs = self.model.forward_outputs(
            images, frequency.to(runtime.device, non_blocking=True)
        )
        losses = self.model.loss_terms(
            outputs, labels, masks.to(runtime.device, non_blocking=True)
        )
        for name, value in losses.items():
            self.component_sums[name] = self.component_sums.get(name, 0.0) + float(
                value.detach()
            ) * len(labels)
        self.component_count += len(labels)
        return losses["overall"]

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.component_sums, self.component_count = {}, 0
        try:
            result = super().train_epoch(
                loader, optimizer, scaler, runtime=runtime, output=output
            )
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            if receipt is None and LRLDataError.marker in str(error):
                try:
                    receipt = json.JSONDecoder().raw_decode(
                        str(error).split(LRLDataError.marker, 1)[1]
                    )[0]
                except (ValueError, TypeError):
                    pass
            atomic_save(
                output / "data-or-training-failure.json",
                receipt or {"error": f"{type(error).__name__}: {error}"},
            )
            raise
        atomic_save(
            output / "loss-components.json",
            {
                "model_images": self.component_count,
                "mean": {
                    k: v / self.component_count for k, v in self.component_sums.items()
                },
                "objective": "similarity_MSE+mask_BCE+CE_equal_weights",
            },
        )
        return result

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if not isinstance(loader.dataset, LRLDataset) or not isinstance(
            loader.collate_fn, LRLCollator
        ):
            raise ValueError(
                "LRL requires synchronized repaired RGB/IDCT/mask transport"
            )
        output = Path(output_dir)
        atomic_save(output / "data-transport.json", loader.dataset.selection_receipt())
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 41,
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
