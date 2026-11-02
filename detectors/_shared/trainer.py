"""Whole-batch image classification with explicit source-validation policy.

The default batch objective is plain CE. Method-owned subclasses may override
compute_loss while retaining receipts, selection and checkpointing. Whole batches
are retained: splitting changes BatchNorm and batch-coupled objectives.
"""

import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn

from data.manifest import sha256
from engine.evaluate import coverage, evaluate
from engine.runtime import atomic_save

logger = logging.getLogger(__name__)


def select_checkpoint(source, destination):
    """Atomically retain an immutable snapshot without serializing it twice.

    Later atomic replacement of last.pth leaves the selected inode unchanged.
    Both paths belong to the same fit directory and filesystem.
    """
    destination = Path(destination)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".checkpoint-"
    ) as directory:
        pending = Path(directory) / "selected.pth"
        os.link(source, pending)
        os.replace(pending, destination)


def integer(settings, key, default, minimum=1):
    value = settings.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def number(settings, key, default, *, minimum=0, strict=False):
    value = settings.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (strict and value == minimum)
    ):
        raise ValueError(
            f"{key} must be finite and {'>' if strict else '>='} {minimum}"
        )
    return value


class _Scorer:
    def __init__(self, model):
        self.model = model

    def predict(self, images, *, runtime):
        with runtime.autocast():
            return self.model(images.to(runtime.device, non_blocking=True)).softmax(1)[
                :, 1
            ]


class ImageClassificationTrainer:
    def __init__(self, model, *, origin=None, objective=None):
        self.model = model
        self.origin = origin or {}
        self.objective = objective if objective is not None else nn.CrossEntropyLoss()

    def compute_loss(self, images, labels):
        logits = self.model(images)
        if logits.shape != (len(labels), 2):
            raise ValueError("Plain classification requires (N, 2) logits")
        return self.objective(logits, labels)

    def make_scorer(self):
        return _Scorer(self.model)

    def method_state_dict(self):
        return None

    def load_method_state_dict(self, state):
        if state is not None:
            raise ValueError("Unexpected method-specific trainer state")

    def build_optimizer(self, settings):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No trainable model parameters")
        options = {
            "lr": number(settings, "learning_rate", 2e-4, strict=True),
            "weight_decay": number(settings, "weight_decay", 5e-4),
        }
        kind = settings.get("optimizer", "adam")
        if kind == "adam":
            betas = settings.get("betas", [0.9, 0.999])
            if len(betas) != 2 or any(
                isinstance(x, bool) or not isinstance(x, (float, int)) or not 0 <= x < 1
                for x in betas
            ):
                raise ValueError("Adam betas must be two values in [0, 1)")
            return torch.optim.Adam(
                parameters,
                **options,
                betas=tuple(betas),
                eps=number(settings, "eps", 1e-8, strict=True),
            )
        if kind == "sgd":
            return torch.optim.SGD(
                parameters, **options, momentum=number(settings, "momentum", 0.9)
            )
        raise ValueError("Classification optimizer must be adam or sgd")

    @staticmethod
    def validate_inputs(loader, validation_loader, monitor):
        if validation_loader is None:
            raise ValueError("Classification training requires fit.validation_input")
        train, val = loader.dataset, validation_loader.dataset
        if train.header["split"] != "train" or val.header["split"] != "val":
            raise ValueError("Training and selection require train/val manifests")
        if train.header["dataset"] != val.header["dataset"]:
            raise ValueError("Selection requires source-domain validation")
        for dataset in (train, val):
            if {r.label for r in dataset.records} != {0, 1}:
                raise ValueError("Training and validation require both classes")

        def identity(r):
            return r.storage, r.path, r.row_group, r.row_index

        if {identity(r) for r in train.records} & {identity(r) for r in val.records}:
            raise ValueError("Train/validation image overlap")
        if {r.video_id for r in train.records if r.video_id} & {
            r.video_id for r in val.records if r.video_id
        }:
            raise ValueError("Train/validation video overlap")
        if monitor == "video.auc" and any(not r.video_id for r in val.records):
            raise ValueError("Video selection requires validation video IDs")

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.model.train()
        expected = {r.id: r for r in loader.dataset.records}
        seen, loss_sum = set(), 0.0
        receipts = output / "samples.partial.jsonl"
        with receipts.open("w") as stream:
            for step, batch in enumerate(loader, 1):
                count = len(batch["labels"])
                if (
                    len(batch["metadata"]) != count
                    or len(batch["pixel_values"]) != count
                ):
                    raise ValueError("Training batch alignment mismatch")
                for metadata, label in zip(
                    batch["metadata"], batch["labels"].tolist(), strict=True
                ):
                    source = expected.get(metadata["id"])
                    if (
                        source is None
                        or source.id in seen
                        or source.label != label
                        or source.dataset != metadata["dataset"]
                        or source.split != metadata["split"]
                    ):
                        raise ValueError("Invalid or repeated training sample identity")
                    seen.add(source.id)
                    stream.write(json.dumps({**metadata, "label": label}) + "\n")
                optimizer.zero_grad(set_to_none=True)
                images = batch["pixel_values"].to(runtime.device, non_blocking=True)
                labels = batch["labels"].to(runtime.device, non_blocking=True)
                with runtime.autocast():
                    loss = self.compute_loss(images, labels)
                if loss.ndim or not torch.isfinite(loss):
                    raise ValueError("Nonfinite or nonscalar classification loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                ):
                    raise ValueError("Nonfinite classification gradient")
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * count
                if step % 100 == 0:
                    logger.info(
                        "Classification batch %d/%d, observed %d, loss %.6f",
                        step,
                        len(loader),
                        len(seen),
                        loss_sum / len(seen),
                    )
        report = coverage(loader.dataset, seen, output)
        if not seen:
            raise ValueError("No images successfully decoded for training")
        receipts.replace(output / "samples.jsonl")
        return loss_sum / len(seen), report

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        monitor = settings.get("monitor", "video.auc")
        if monitor not in ("image.auc", "video.auc"):
            raise ValueError("Classification monitor must be image.auc or video.auc")
        self.validate_inputs(loader, validation_loader, monitor)
        if settings.get("microbatch_size") is not None:
            raise ValueError(
                "This trainer uses whole batches; microbatching changes BatchNorm semantics"
            )
        epochs = integer(settings, "epochs", 10)
        patience = settings.get("patience")
        if patience is not None:
            patience = integer(settings, "patience", None)
        delta = number(settings, "min_delta", 0.0)
        self.model.to(runtime.device)
        self.objective.to(runtime.device)
        optimizer = self.build_optimizer(settings)
        scaler = torch.amp.GradScaler("cuda", enabled=runtime.precision == "fp16")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        protocol = {
            "id": "remake-source-validation-v1",
            "origin": self.origin,
            "settings": settings,
            "monitor": monitor,
            "native_batch_semantics": "whole_batch",
            "torch_initial_seed": torch.initial_seed(),
            "batch_size": loader.batch_size,
            "loader_initial_seed": loader.generator.initial_seed(),
            "inputs": {
                name: {
                    "path": str(data.manifest_path),
                    "sha256": sha256(data.manifest_path),
                    "samples": len(data),
                }
                for name, data in (
                    ("train", loader.dataset),
                    ("validation", validation_loader.dataset),
                )
            },
            "note": "Source validation is the remake selection policy; not a claim of original full-protocol reproduction",
        }
        atomic_save(output / "protocol.json", protocol)
        atomic_save(
            output / "environment.json",
            {
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in (
                        "torch",
                        "transformers",
                        "numpy",
                        "opencv-python-headless",
                        "albumentations",
                    )
                },
                "precision": runtime.precision,
                "device": str(runtime.device),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "exact_training_resume_supported": False,
            },
        )
        history, stale, best, best_epoch = [], 0, -math.inf, None
        for epoch in range(1, epochs + 1):
            epoch_dir = output / f"epoch-{epoch:03d}"
            training = epoch_dir / "train"
            training.mkdir(parents=True)
            loss, train_report = self.train_epoch(
                loader, optimizer, scaler, runtime=runtime, output=training
            )
            # Validation and serialized snapshots do not use parameter gradients.
            # The next batch also starts with zero_grad; discard these tensors
            # before the large model/optimizer checkpoint write.
            optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            validation = evaluate(
                self.make_scorer(),
                validation_loader,
                runtime=runtime,
                settings={
                    "threshold": 0.5,
                    "video_aggregation": "mean",
                    "group_by": ["category"],
                },
                output_dir=epoch_dir / "validation",
            )
            unit, metric = monitor.split(".")
            value = validation[unit][metric] if validation[unit] is not None else None
            if value is None or not math.isfinite(value):
                raise ValueError("Source validation selection metric is undefined")
            improved = value > best + delta
            if improved:
                best, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            record = {
                "epoch": epoch,
                "loss": loss,
                "train": train_report,
                "validation": validation,
                "improved": improved,
                "stale": stale,
            }
            history.append(record)
            atomic_save(output / "history.json", history)
            numpy_rng = np.random.get_state()
            snapshot = {
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "objective": self.objective.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "record": record,
                "protocol": protocol,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                "cuda_rng": torch.cuda.get_rng_state_all()
                if runtime.device.type == "cuda"
                else [],
                "loader_rng": loader.generator.get_state(),
            }
            method_state = self.method_state_dict()
            if method_state is not None:
                snapshot["method_state"] = method_state
            atomic_save(output / "last.pth", snapshot, torch_format=True)
            if improved:
                select_checkpoint(output / "last.pth", output / "best.pth")
            del snapshot
            logger.info(
                "Classification epoch %d loss %.6f %s %.6f", epoch, loss, monitor, value
            )
            if patience is not None and stale >= patience:
                break
        del optimizer, scaler
        selected = torch.load(
            output / "best.pth", weights_only=True, mmap=True, map_location="cpu"
        )
        self.model.load_state_dict(selected["state_dict"], strict=True)
        self.objective.load_state_dict(selected["objective"], strict=True)
        self.load_method_state_dict(selected.get("method_state"))
        self.model.eval()
        return {
            "epochs": len(history),
            "best_epoch": best_epoch,
            "best_validation_value": best,
            "monitor": monitor,
            "stopped_early": len(history) < epochs,
            "history": "history.json",
            "train": history[-1]["train"],
        }
