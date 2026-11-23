"""Effort optimization and source-validation selection; no target-based tuning."""

import importlib.metadata
import json
import logging
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from data.manifest import sha256
from engine.accelerator import cuda_metadata
from engine.training_state import TrainingSnapshot
from engine.evaluate import coverage, evaluate
from engine.runtime import atomic_save

logger = logging.getLogger(__name__)


class _Scorer:
    def __init__(self, model):
        self.model = model

    def predict(self, images, *, runtime):
        with runtime.autocast():
            return self.model(images.to(runtime.device, non_blocking=True)).softmax(1)[
                :, 1
            ]


def positive_integer(settings, name, default):
    value = settings.get(name, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"Effort {name} must be an integer >= 1")
    return value


def finite_number(settings, name, default, *, zero_allowed=False):
    value = settings.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero_allowed)
    ):
        raise ValueError(
            f"Effort {name} must be finite and {'nonnegative' if zero_allowed else 'positive'}"
        )
    return value


class EffortTrainer:
    def __init__(self, model, *, origin=None):
        self.model = model
        self.origin = origin or {}

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if validation_loader is None:
            raise ValueError("Effort training requires fit.validation_input")
        train, validation = loader.dataset, validation_loader.dataset
        if train.header["split"] != "train" or validation.header["split"] != "val":
            raise ValueError("Effort requires train and val manifests")
        if train.header["dataset"] != validation.header["dataset"]:
            raise ValueError(
                "Effort model selection requires a source-domain validation set"
            )
        if any(
            {r.label for r in data.records} != {0, 1} for data in (train, validation)
        ):
            raise ValueError("Effort training and validation each require both classes")

        def identity(record):
            return (record.storage, record.path, record.row_group, record.row_index)

        if {identity(r) for r in train.records} & {
            identity(r) for r in validation.records
        }:
            raise ValueError("Effort train/validation image overlap")
        train_videos = {r.video_id for r in train.records if r.video_id}
        if train_videos & {r.video_id for r in validation.records if r.video_id}:
            raise ValueError("Effort train/validation video overlap")
        monitor = settings.get("monitor", "video.auc")
        if monitor not in ("image.auc", "video.auc"):
            raise ValueError("Effort monitor must be image.auc or video.auc")
        selection = (
            "source_validation."
            + monitor
            + (".mean_probability" if monitor == "video.auc" else "")
        )
        if monitor == "video.auc" and any(not r.video_id for r in validation.records):
            raise ValueError("Effort video AUC selection requires validation video IDs")
        epochs = positive_integer(settings, "epochs", 10)
        patience = positive_integer(settings, "patience", 2)
        microbatch = positive_integer(settings, "microbatch_size", 8)
        delta = finite_number(settings, "min_delta", 1e-4, zero_allowed=True)
        lr = finite_number(settings, "learning_rate", 2e-4)
        decay = finite_number(settings, "weight_decay", 5e-4, zero_allowed=True)
        # The baseline deliberately fixes Adam betas/epsilon, no scheduler or extra losses.
        self.model.to(runtime.device)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=decay,
            amsgrad=False,
        )
        scaler = runtime.grad_scaler()
        training_snapshot = TrainingSnapshot(
            settings, loader, validation_loader, runtime
        )
        training_snapshot.restore(self, optimizer, scaler)
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        protocol = {
            "origin": self.origin,
            "settings": settings,
            "torch_initial_seed": torch.initial_seed(),
            "loader_initial_seed": loader.generator.initial_seed()
            if loader.generator is not None
            else None,
            "inputs": {
                name: {
                    "manifest": str(dataset.manifest_path),
                    "sha256": sha256(dataset.manifest_path),
                    "samples": len(dataset),
                    "dataset": dataset.header["dataset"],
                    "split": dataset.header["split"],
                }
                for name, dataset in (("train", train), ("validation", validation))
            },
        }
        atomic_save(root / "protocol.json", protocol)
        atomic_save(
            root / "environment.json",
            {
                "packages": {
                    k: importlib.metadata.version(k)
                    for k in (
                        "torch",
                        "transformers",
                        "albumentations",
                        "numpy",
                        "opencv-python-headless",
                    )
                },
                "device": str(runtime.device),
                "precision": runtime.precision,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                **cuda_metadata(runtime.device),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "objective": "cross_entropy",
                "optimizer": "Adam",
                "betas": [0.9, 0.999],
                "learning_rate": lr,
                "weight_decay": decay,
                "eps": 1e-8,
                "batch_size": loader.batch_size,
                "microbatch_size": microbatch,
                "selection": selection,
                "exact_training_resume_supported": training_snapshot.resumable,
                "resume_scope": "selected epoch"
                if training_snapshot.resumable
                else None,
            },
        )
        best, stale, best_epoch = -math.inf, 0, None
        history = []
        expected = {r.id: r for r in train.records}
        for epoch in range(1, epochs + 1):
            epoch_dir = root / f"epoch-{epoch:03d}"
            train_dir = epoch_dir / "train"
            train_dir.mkdir(parents=True)
            self.model.train()
            seen, loss_sum = set(), 0.0
            receipts = train_dir / "samples.partial.jsonl"
            with receipts.open("w") as stream:
                for batch in loader:
                    size = len(batch["labels"])
                    if (
                        len(batch["metadata"]) != size
                        or len(batch["pixel_values"]) != size
                    ):
                        raise ValueError("Effort training batch alignment mismatch")
                    for metadata, label in zip(
                        batch["metadata"], batch["labels"].tolist(), strict=True
                    ):
                        sample_id = metadata["id"]
                        source = expected.get(sample_id)
                        if (
                            sample_id in seen
                            or source is None
                            or source.label != label
                            or source.dataset != metadata["dataset"]
                            or source.split != metadata["split"]
                        ):
                            raise ValueError(
                                f"Invalid/duplicate Effort training identity: {sample_id}"
                            )
                        seen.add(sample_id)
                        stream.write(json.dumps({**metadata, "label": label}) + "\n")
                    with runtime.time_region("fit/batch", samples=size):
                        optimizer.zero_grad(set_to_none=True)
                        for start in range(0, size, microbatch):
                            images = batch["pixel_values"][
                                start : start + microbatch
                            ].to(runtime.device, non_blocking=True)
                            labels = batch["labels"][start : start + microbatch].to(
                                runtime.device, non_blocking=True
                            )
                            with runtime.autocast():
                                logits = self.model(images)
                                loss = F.cross_entropy(logits, labels)
                            if not torch.isfinite(loss):
                                raise ValueError("Nonfinite Effort training loss")
                            scaler.scale(loss * len(labels) / size).backward()
                            loss_sum += float(loss.detach()) * len(labels)
                        scaler.unscale_(optimizer)
                        if any(
                            p.grad is not None and not torch.isfinite(p.grad).all()
                            for p in parameters
                        ):
                            raise ValueError("Nonfinite Effort training gradient")
                        scaler.step(optimizer)
                        scaler.update()
            train_report = coverage(train, seen, train_dir)
            if not seen:
                raise ValueError("No images successfully decoded for Effort training")
            receipts.replace(train_dir / "samples.jsonl")
            self.model.eval()
            validation_report = evaluate(
                _Scorer(self.model),
                validation_loader,
                runtime=runtime,
                settings={
                    "threshold": 0.5,
                    "video_aggregation": "mean",
                    "group_by": ["category"],
                },
                output_dir=epoch_dir / "validation",
                phase="fit/validation",
            )
            unit, metric = monitor.split(".")
            values = validation_report[unit]
            auc = values[metric] if values is not None else None
            if auc is None or not math.isfinite(auc):
                raise ValueError(
                    f"Source validation {monitor} is undefined after decoding"
                )
            improved = auc > best + delta
            if improved:
                best, stale, best_epoch = auc, 0, epoch
            else:
                stale += 1
            record = {
                "epoch": epoch,
                "loss": loss_sum / len(seen),
                "train": train_report,
                "validation": validation_report,
                "improved": improved,
                "stale": stale,
            }
            history.append(record)
            atomic_save(root / "history.json", history)
            checkpoint = {
                **training_snapshot.capture(self, optimizer, scaler),
                "epoch": epoch,
                "record": record,
                "protocol": protocol,
            }
            atomic_save(root / "last.pth", checkpoint, torch_format=True)
            if improved:
                atomic_save(root / "best.pth", checkpoint, torch_format=True)
            del checkpoint
            logger.info(
                "Effort epoch %d: loss %.6f, validation %s %.6f, stale %d",
                epoch,
                record["loss"],
                monitor,
                auc,
                stale,
            )
            if stale >= patience:
                break
        # Return the selected model, not the final optimization iterate.
        saved = torch.load(
            root / "best.pth", weights_only=True, map_location="cpu", mmap=True
        )
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()
        return {
            "epochs": len(history),
            "best_epoch": best_epoch,
            "monitor": monitor,
            "best_validation_auc": best,
            "training_checkpoint_id": training_snapshot.id
            if training_snapshot.resumable
            else None,
            **(
                {"best_validation_video_auc": best}
                if monitor == "video.auc"
                else {"best_validation_image_auc": best}
            ),
            "stopped_early": len(history) < epochs,
            "history": "history.json",
            "selection": selection,
            "train": history[-1]["train"],
        }
