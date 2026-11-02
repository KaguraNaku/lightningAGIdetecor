"""Whole generated pairs with aligned raw-boundary supervision and provenance."""

import json
from pathlib import Path

import torch

from detectors._shared.paired import PairedImageTrainer
from detectors.fwa.generation import pixel_digest
from engine.runtime import atomic_save
from .dataset import FaceXrayDataset, FaceXrayPairCollator, FaceXrayGenerationError


class FaceXrayTrainer(PairedImageTrainer):
    fallback_statistic = "generation_replacements"

    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin)
        self.component_sums, self.component_count = {}, 0

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        masks = batch.get("masks")
        if (
            not isinstance(masks, torch.Tensor)
            or masks.shape != (len(labels), 1, 256, 256)
            or masks.dtype != torch.float64
            or not torch.isfinite(masks).all()
        ):
            raise ValueError(
                "FaceXray training requires a finite aligned256 boundary for every model view"
            )
        sources = {item["metadata"]["id"]: item for item in batch["sources"]}
        for mask, metadata in zip(masks, batch["metadata"], strict=True):
            if metadata["view"] == "original":
                if torch.count_nonzero(mask):
                    raise ValueError(
                        "FaceXray original view must retain its zero boundary"
                    )
            elif (
                pixel_digest(mask.numpy())
                != sources[metadata["id"]]["generation"]["boundary_sha256"]
            ):
                raise ValueError(
                    "FaceXray generated boundary differs from its parent receipt"
                )
        losses = self.model.loss_terms(
            self.model.forward_outputs(images),
            labels,
            mask=masks.to(runtime.device, non_blocking=True),
        )
        for key, value in losses.items():
            self.component_sums[key] = self.component_sums.get(key, 0.0) + float(
                value.detach()
            ) * len(labels)
        self.component_count += len(labels)
        return losses["overall"]

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.component_sums, self.component_count = {}, 0
        loss, coverage = super().train_epoch(
            loader, optimizer, scaler, runtime=runtime, output=output
        )
        atomic_save(
            output / "loss-components.json",
            {
                "model_images": self.component_count,
                "mean": {
                    key: value / self.component_count
                    for key, value in self.component_sums.items()
                },
                "objective": "CE+1000*raw_mask_MSE_float32; means over complete effective2B views",
            },
        )
        return loss, coverage

    def training_batches(self, loader, output):
        try:
            yield from loader
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            marker = FaceXrayGenerationError.marker
            if receipt is None and marker in str(error):
                try:
                    receipt = json.JSONDecoder().raw_decode(
                        str(error).split(marker, 1)[1]
                    )[0]
                except (ValueError, TypeError):
                    receipt = {"error": str(error)}
            atomic_save(
                output / "generation-failure.json",
                receipt or {"error": f"{type(error).__name__}: {error}"},
            )
            raise

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if not isinstance(loader.dataset, FaceXrayDataset) or not isinstance(
            loader.collate_fn, FaceXrayPairCollator
        ):
            raise ValueError(
                "FaceXray fitting requires its bound bank dataset and aligned pair collator"
            )
        output = Path(output_dir)
        atomic_save(
            output / "source-selection.json", loader.dataset.selection_receipt()
        )
        atomic_save(
            output / "paired-protocol.json",
            {
                "source_batch_size": loader.batch_size,
                "model_batch_size": 2 * loader.batch_size,
                "source_labels": "Mixed real0/fake1 backgrounds; every generated view has label1",
                "order": "All fake then all real, jointly shuffled with masks/labels/parent identities",
                "mask": "Raw float64 boundary transported, float32 source MSE with weight1000; zero targets for original views",
                "neighbor_policy": loader.dataset.bank.header["origin"].get(
                    "neighbor_policy"
                ),
                "scheduler": None,
            },
        )
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 11,
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
