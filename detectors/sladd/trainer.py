"""Whole-pair SLADD epochs, native validation cadence and two Adam phases."""

import gzip
import json
import math
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from detectors._shared.trainer import (
    ImageClassificationTrainer,
    _Scorer,
    integer,
    number,
    select_checkpoint,
)
from engine.evaluate import coverage, evaluate
from engine.runtime import atomic_save
from engine.saved_tensors import DiskSavedTensors
from .dataset import metadata, SLADDPairDataset
from .images import SLADDImageDataset
from .optimization import SLADDOptimization
from .resources import native_saved_graph


def validation_interval(batches, epoch):
    interval = batches // (1 if epoch == 0 else 2)
    if interval < 1:
        raise ValueError(
            "Native SLADD validation cadence is undefined for <2 batches after epoch0; no silent cadence repair"
        )
    return interval


class SLADDTrainer:
    def __init__(self, model, *, data_profile, origin=None, on_start=None):
        self.model, self.data_profile = model, data_profile
        self.origin, self.on_start = origin or {}, on_start

    def validate_batch(self, batch, dataset, seen):
        """Check the bound source identities before in-place pair switching."""
        count = len(batch["pairs"])
        if (
            not 1 <= count <= 32
            or batch["pixel_values"].shape != (2 * count, 3, 256, 256)
            or batch["pixel_values"].dtype != torch.float32
            or batch["labels"].dtype != torch.int64
            or batch["labels"].tolist() != [0] * count + [1] * count
            or batch["landmarks"].shape != (2 * count, 81, 2)
            or batch["masks"].shape != (2 * count, 256, 256, 3)
            or len(batch["metadata"]) != 2 * count
        ):
            raise ValueError("SLADD requires aligned native real-all/fake-all pairs")
        fake = {row.id: row for row in dataset.records}
        real = {row.id: row for row in dataset.real_records}
        pending = set()
        for index, pair in enumerate(batch["pairs"]):
            identifier = pair["indexed_fake_id"]
            if (
                identifier not in fake
                or identifier in seen
                or identifier in pending
                or pair["real_id"] not in real
                or pair["profile"] != self.data_profile
            ):
                raise ValueError("Unknown/repeated SLADD indexed fake or real identity")
            for role, row, offset in (
                ("real", real[pair["real_id"]], 0),
                ("fake", fake[identifier], count),
            ):
                expected = {
                    **metadata(row),
                    "pair_index": index,
                    "view": role,
                    "pair_id": identifier,
                }
                if batch["metadata"][index + offset] != expected:
                    raise ValueError("SLADD collated metadata differs from bound pair")
                if pair["bindings"][role] != dataset.plan.bindings[row.id]:
                    raise ValueError("SLADD collated bindings differ from source plan")
            pending.add(identifier)
        seen.update(pending)

    def validate_inputs(self, loader, validation_loader, monitor):
        if (
            not isinstance(loader.dataset, SLADDPairDataset)
            or loader.dataset.plan.profile != self.data_profile
        ):
            raise ValueError("SLADD training requires its bound paired source dataset")
        if (
            not isinstance(
                getattr(validation_loader, "dataset", None), SLADDImageDataset
            )
            or validation_loader.dataset.stage != "validation"
        ):
            raise ValueError("SLADD requires source-domain validation images")
        train, val = loader.dataset, validation_loader.dataset
        proxy = SimpleNamespace(
            dataset=SimpleNamespace(
                header=train.header, records=train.plan.source_records
            )
        )
        ImageClassificationTrainer.validate_inputs(proxy, validation_loader, monitor)
        if train.root != val.root or {r.id for r in train.plan.source_records} & {
            r.id for r in val.records
        }:
            raise ValueError(
                "SLADD complete train/validation source identities or roots differ"
            )
        if self.data_profile != "source_lmdb_rgb_bytes":
            raise ValueError(
                "Formal source filesystem-gray masks fail native synthesis; select the explicit LMDB byte-format override for fitting"
            )
        for current in (loader, validation_loader):
            if (
                current.batch_size != 32
                or current.drop_last
                or current.generator is None
            ):
                raise ValueError(
                    "Selected SLADD loaders require batch32, retained tail and explicit RNG generator"
                )
            if (
                current.num_workers > 1
                or current.num_workers
                and current.prefetch_factor > 2
            ):
                raise ValueError(
                    "Bounded SLADD profile supports at most one worker and prefetch2; original8-worker RNG is unverified"
                )

    @staticmethod
    def optimizer_settings(settings):
        if (
            settings.get("optimizer", "adam") != "adam"
            or settings.get("patience") is not None
            or settings.get("lr_scheduler") is not None
            or settings.get("microbatch_size") is not None
            or settings.get("min_delta", 0) != 0
        ):
            raise ValueError(
                "Native SLADD uses Adam, no scheduler/early-stop/microbatch or selection delta"
            )
        betas = settings.get("betas", [0.9, 0.999])
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1
                for v in betas
            )
        ):
            raise ValueError("SLADD Adam betas must be two values in [0,1)")
        return {
            "learning_rate": number(settings, "learning_rate", 2e-4, strict=True),
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": number(settings, "eps", 1e-8, strict=True),
            "weight_decay": number(settings, "weight_decay", 5e-4),
        }

    def preflight(self, loader, validation_loader, *, runtime, settings, output):
        allowed = {
            "input",
            "validation_input",
            "epochs",
            "monitor",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "betas",
            "eps",
            "patience",
            "min_delta",
            "lr_scheduler",
            "microbatch_size",
            "disk_saved_tensors",
            "saved_activation_budget_gib",
            "disk_reserve_gib",
        }
        if set(settings) - allowed:
            raise ValueError("Unknown SLADD fit settings")
        if (
            runtime.device.type != "cpu"
            or runtime.precision != "fp32"
            or torch.get_default_dtype() != torch.float32
        ):
            raise ValueError(
                "Selected SLADD fit profile is CPU fp32; GPU/AMP fit is not validated"
            )
        if settings.get("disk_saved_tensors", True) is not True:
            raise ValueError(
                "SLADD CPU fitting requires exact saved-tensor disk transport"
            )
        monitor = settings.get("monitor", "image.auc")
        if monitor not in {"image.auc", "video.auc"}:
            raise ValueError("SLADD selection requires declared image.auc or video.auc")
        self.validate_inputs(loader, validation_loader, monitor)
        options = self.optimizer_settings(settings)
        epochs = integer(settings, "epochs", 41)
        for epoch in range(epochs):
            validation_interval(len(loader), epoch)
        graph = native_saved_graph()
        self.reserve = integer(settings, "disk_reserve_gib", 8, minimum=8) * 1024**3
        self.checkpoint_headroom = 2 * 1024**3
        required = (
            math.ceil(graph["logical_nonpersistent_saved_bytes"] * 1.25)
            * len(loader)
            * epochs
        )
        self.activation_budget = (
            integer(
                settings, "saved_activation_budget_gib", math.ceil(required / 1024**3)
            )
            * 1024**3
        )
        if self.activation_budget < required:
            raise ValueError(
                "SLADD activation budget must cover every whole-batch graph in the declared run"
            )
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(
                "SLADD fit output must be fresh; existing runs are preserved"
            )
        protocol = {
            "method": "sladd",
            "origin": self.origin,
            "requested_settings": settings,
            "epochs": epochs,
            "native_epochs": list(range(epochs)),
            "native_default_epochs": "0..40 inclusive41",
            "optimizer": options,
            "policy_optimizer": {
                "learning_rate": options["learning_rate"] / 4,
                "beta1": options["beta1"] / 4,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": options["weight_decay"],
            },
            "monitor": monitor,
            "validation": "Source val only;native global-step modulus with floor(len(loader)/1 at epoch0,/2 later). Not source multi-dataset selection or original8-worker global RNG.",
            "exact_resume_supported": False,
            "torch_initial_seed": torch.initial_seed(),
            "train_loader_initial_seed": loader.generator.initial_seed(),
            "validation_loader_initial_seed": validation_loader.generator.initial_seed(),
            "train": loader.dataset.selection_receipt(),
            "validation_data": validation_loader.dataset.selection_receipt(),
            "symbolic_saved_graph": graph,
            "retained_activation_budget_bytes": self.activation_budget,
            "disk_reserve_bytes": self.reserve,
            "checkpoint_headroom_bytes": self.checkpoint_headroom,
        }
        atomic_save(output / "source-data-protocol.json", protocol)
        free = shutil.disk_usage(output).free
        passed = (
            free >= self.activation_budget + self.reserve + self.checkpoint_headroom
        )
        atomic_save(
            output / "resource-preflight.json",
            {
                "passed": passed,
                "free_bytes": free,
                "required_bytes": self.activation_budget
                + self.reserve
                + self.checkpoint_headroom,
                "training_started": False,
                "symbolic_estimate_is_not_measured_peak": True,
            },
        )
        if not passed:
            raise OSError(
                "SLADD retained full run plus disk/checkpoint reserves cannot fit; no training started"
            )
        return epochs, monitor, options

    def _snapshot(self, core, output, *, epoch, step, event):
        core.detector_optimizer.zero_grad(set_to_none=True)
        core.synthesizer_optimizer.zero_grad(set_to_none=True)
        numpy = np.random.get_state()
        atomic_save(
            output / "last.pth",
            {
                "state_dict": self.model.state_dict(),
                "detector_optimizer": core.detector_optimizer.state_dict(),
                "synthesizer_optimizer": core.synthesizer_optimizer.state_dict(),
                "native_epoch": epoch,
                "step": step,
                "event": event,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": [numpy[0], numpy[1].tolist(), *numpy[2:]],
                "training_loader_rng": self.train_loader.generator.get_state(),
                "validation_loader_rng": self.val_loader.generator.get_state(),
                "exact_resume_supported": False,
            },
            torch_format=True,
        )

    def _step(self, batch, core, directory, step):
        store = DiskSavedTensors(
            self.model, directory / f"batch-{step:06d}-saved-tensors"
        )
        original = store.pack
        before = self.activation_bytes

        def bounded_pack(tensor):
            storage = tensor.untyped_storage()
            key = StorageWeakRef(storage).cdata, tensor._version, str(tensor.dtype)
            if (
                storage.data_ptr() not in store.persistent
                and storage.nbytes() >= store.threshold
                and key not in store.cache
            ):
                possible = storage.nbytes() + 256 * 1024
                written = sum(row["bytes"] for row in store.files)
                if (
                    before + written + possible > self.activation_budget
                    or shutil.disk_usage(directory).free - possible
                    < self.reserve + self.checkpoint_headroom
                ):
                    raise OSError(
                        "SLADD saved activation would cross retained-run or disk reserve budget"
                    )
            return original(tensor)

        store.pack = bounded_pack
        try:
            core.begin_batch()
            with store:
                outputs = self.model.forward_outputs(
                    batch["pixel_values"],
                    labels=batch["labels"],
                    landmarks=batch["landmarks"],
                    masks=batch["masks"],
                    inference=False,
                )
                losses = core.step(outputs)
            return outputs, losses
        finally:
            receipt = store.report()
            self.activation_bytes += receipt["bytes_written"]
            atomic_save(directory / f"batch-{step:06d}-saved-tensors.json", receipt)

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        output = Path(output_dir)
        epochs, monitor, options = self.preflight(
            loader, validation_loader, runtime=runtime, settings=settings, output=output
        )
        self.train_loader, self.val_loader = loader, validation_loader
        self.activation_bytes = 0
        self.model.to(runtime.device)
        if any(not p.requires_grad for p in self.model.parameters()):
            raise ValueError("Selected SLADD optimizes every main/policy parameter")
        core = SLADDOptimization(
            self.model.detector_parameters(), self.model.synthesizer, **options
        )
        if self.on_start is not None:
            self.on_start()
        atomic_save(
            output / "environment.json",
            {
                "torch": str(torch.__version__),
                "device": str(runtime.device),
                "precision": runtime.precision,
                "mkldnn": torch.backends.mkldnn.enabled,
                "source_step_math": "policy update before detector backward;generated labels;native pairwise magnitude broadcast",
                "exact_resume_supported": False,
            },
        )
        history, best, selected = [], -math.inf, None
        try:
            for epoch in range(epochs):
                directory = output / f"epoch-{epoch:03d}"
                directory.mkdir()
                seen, loss_sum = set(), 0.0
                interval = validation_interval(len(loader), epoch)
                with gzip.open(
                    directory / "batches.partial.jsonl.gz", "wt", compresslevel=1
                ) as stream:
                    for step, batch in enumerate(loader, 1):
                        self.model.train()
                        self.validate_batch(batch, loader.dataset, seen)
                        numpy = np.random.get_state()
                        before = {
                            "torch": torch.get_rng_state().tolist(),
                            "python": random.getstate(),
                            "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
                        }
                        outputs, losses = self._step(batch, core, directory, step)
                        scalar = {
                            key: float(value.detach())
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in losses.items()
                        }
                        loss_sum += scalar["overall"]
                        stream.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "pairs": batch["pairs"],
                                    "metadata": batch["metadata"],
                                    "rng_before_model": before,
                                    "switched_source_labels": batch["labels"].tolist(),
                                    "generated_labels": outputs["label"].tolist(),
                                    "type_labels": outputs["type_label"].tolist(),
                                    "magnitude_mask": outputs["mag_mask"].tolist(),
                                    "losses": scalar,
                                }
                            )
                            + "\n"
                        )
                        del outputs, losses
                        if (epoch * len(loader) + step) % interval == 0:
                            self.model.eval()
                            validation = evaluate(
                                _Scorer(self.model),
                                validation_loader,
                                runtime=runtime,
                                settings={
                                    "threshold": 0.5,
                                    "video_aggregation": "mean",
                                    "group_by": ["category"],
                                },
                                output_dir=directory / f"validation-{step:06d}",
                            )
                            unit, metric = monitor.split(".")
                            value = (
                                validation[unit][metric] if validation[unit] else None
                            )
                            if value is None or not math.isfinite(value):
                                raise ValueError(
                                    "SLADD source-validation selection metric is undefined"
                                )
                            improved = value > best
                            event = {
                                "native_epoch": epoch,
                                "step": step,
                                "validation": validation,
                                "improved": improved,
                            }
                            history.append(event)
                            atomic_save(output / "history.json", history)
                            self._snapshot(
                                core, output, epoch=epoch, step=step, event=event
                            )
                            if improved:
                                best, selected = (
                                    value,
                                    {"native_epoch": epoch, "step": step},
                                )
                                select_checkpoint(
                                    output / "last.pth", output / "best.pth"
                                )
                (directory / "batches.partial.jsonl.gz").replace(
                    directory / "batches.jsonl.gz"
                )
                report = coverage(loader.dataset, seen, directory)
                if report["skipped"] or not seen:
                    raise ValueError(
                        "SLADD epoch did not consume every indexed fake source"
                    )
                atomic_save(
                    directory / "training.json",
                    {"coverage": report, "mean_batch_loss": loss_sum / len(loader)},
                )
                self._snapshot(
                    core,
                    output,
                    epoch=epoch,
                    step=len(loader),
                    event={"kind": "epoch_end"},
                )
        except Exception as error:
            atomic_save(
                output / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "training_started": True,
                    "optimizer_state_may_be_partial": True,
                },
            )
            raise
        saved = torch.load(
            output / "best.pth", map_location="cpu", weights_only=True, mmap=True
        )
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()
        return {
            "epochs": epochs,
            "selected": selected,
            "best_validation_value": best,
            "monitor": monitor,
            "source_only_selection": True,
            "exact_resume_supported": False,
            "protocol": "source-data-protocol.json",
            "history": "history.json",
            "retained_activation_bytes": self.activation_bytes,
        }
