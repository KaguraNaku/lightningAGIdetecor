"""Native whole-batch ProDet: ordinary CUDA autograd or retained CPU audit activations."""

from contextlib import contextmanager
from pathlib import Path
import math
import shutil

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from detectors._shared.trainer import ImageClassificationTrainer, integer
from engine.runtime import atomic_save
from engine.saved_tensors import DiskSavedTensors
from .dataset import ProDetDataset
from .images import ProDetImageDataset
from .protocol import validate_source_splits


class ProDetTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, data_profile, origin=None):
        super().__init__(model, origin=origin)
        if data_profile not in {"native_entrypoint", "specialized_quadruples"}:
            raise ValueError("ProDet Trainer requires an explicit data profile")
        self.data_profile, self.native_epoch = data_profile, -1

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        if self.native_epoch < 0:
            raise ValueError("ProDet loss requires a zero-based active native epoch")
        if self.data_profile == "specialized_quadruples":
            targets = batch["prodet_labels"]
            count = len(images)
            expected = torch.tensor([0.0, 1.0, 3.0, 2.0]).repeat_interleave(count // 4)
            if (
                targets.dtype != torch.float32
                or count % 4
                or not torch.equal(targets.cpu(), expected)
            ):
                raise ValueError(
                    "ProDet specialized targets must retain ordered float0/1/3/2 domains"
                )
        else:
            targets = labels
            if (
                targets.dtype != torch.int64
                or not ((targets == 0) | (targets == 1)).all()
            ):
                raise ValueError(
                    "ProDet native entrypoint retains generic int64 binary targets"
                )
        outputs = self.model.forward_outputs(
            images, inference=False, epoch=self.native_epoch
        )
        return self.model.loss_terms(outputs, targets.to(runtime.device))["overall"]

    def train_epoch(self, *args, **kwargs):
        self.native_epoch += 1
        return super().train_epoch(*args, **kwargs)

    def method_state_dict(self):
        return {"data_profile": self.data_profile, "native_epoch": self.native_epoch}

    def load_method_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or set(state) != {"data_profile", "native_epoch"}
            or state["data_profile"] != self.data_profile
            or type(state["native_epoch"]) is not int
            or state["native_epoch"] < -1
        ):
            raise ValueError("Invalid ProDet training epoch state")
        self.native_epoch = state["native_epoch"]

    def build_optimizer(self, settings):
        if settings.get("optimizer", "adam") != "adam" or any(
            not p.requires_grad for p in self.model.parameters()
        ):
            raise ValueError("Native ProDet uses Adam over every model parameter")
        return super().build_optimizer(settings)

    def validate_inputs(
        self, loader, validation_loader, monitor, *, training_labels=None
    ):
        super().validate_inputs(loader, validation_loader, monitor)
        if (
            not isinstance(validation_loader.dataset, ProDetImageDataset)
            or validation_loader.dataset.stage != "validation"
        ):
            raise ValueError(
                "ProDet validation requires generic native images in validation role"
            )
        if self.data_profile == "specialized_quadruples":
            if not isinstance(loader.dataset, ProDetDataset):
                raise ValueError(
                    "ProDet specialized training requires complete quadruple bindings"
                )
            validate_source_splits(loader.dataset.plan, validation_loader.dataset)
        elif (
            not isinstance(loader.dataset, ProDetImageDataset)
            or loader.dataset.stage != "train"
        ):
            raise ValueError("ProDet native entrypoint requires generic train images")
        if loader.dataset.root.resolve() != validation_loader.dataset.root.resolve():
            raise ValueError("ProDet source train/validation roots must agree")
        if {row.id for row in loader.dataset.source_records} & {
            row.id for row in validation_loader.dataset.records
        }:
            raise ValueError("ProDet complete-source train/validation id overlap")

    @contextmanager
    def batch_context(self, batch, *, runtime, output, step):
        if self.data_profile == "specialized_quadruples":
            receipts = batch["generation_receipts"]
            if len(receipts) * 4 != len(batch["labels"]):
                raise ValueError(
                    "ProDet generation receipts must cover every quadruple"
                )
            atomic_save(
                Path(output) / f"batch-{step:06d}-generation.json",
                {
                    "native_epoch": self.native_epoch,
                    "sources": receipts,
                },
            )
        if runtime.device.type == "cuda":
            # Ordinary GPU autograd owns activations; never retain audit tensors on disk.
            yield
            return
        store = DiskSavedTensors(
            self.model, Path(output) / f"batch-{step:06d}-saved-tensors"
        )
        original_pack, before = store.pack, self.activation_bytes

        def pack(tensor):
            storage = tensor.untyped_storage()
            key = StorageWeakRef(storage).cdata, tensor._version, str(tensor.dtype)
            if (
                storage.data_ptr() not in store.persistent
                and storage.nbytes() >= store.threshold
                and key not in store.cache
            ):
                possible = storage.nbytes() + 256 * 1024
                written = sum(row["bytes"] for row in store.files)
                if before + written + possible > self.activation_budget:
                    raise OSError(
                        "ProDet retained activation budget would be exceeded;no artifacts removed"
                    )
                if (
                    shutil.disk_usage(output).free - possible
                    < self.disk_reserve + self.checkpoint_headroom
                ):
                    raise OSError(
                        "ProDet activation write would cross disk/checkpoint reserve"
                    )
            return original_pack(tensor)

        store.pack = pack
        try:
            with store:
                yield
        finally:
            receipt = store.report()
            self.activation_bytes = before + receipt["bytes_written"]
            receipt.update(
                native_epoch=self.native_epoch,
                cumulative_bytes=self.activation_bytes,
                budget_bytes=self.activation_budget,
                retention="Keep on success and failure",
            )
            atomic_save(Path(output) / f"batch-{step:06d}-saved-tensors.json", receipt)

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if self.native_epoch != -1:
            raise ValueError("Create a fresh ProDet Trainer for each fit")
        if runtime.device.type not in {"cpu", "cuda"} or runtime.precision != "fp32":
            raise ValueError(
                "ProDet fitting supports CPU audit or CUDA FP32;AMP remains unverified"
            )
        if (
            loader.batch_size != 24
            or validation_loader is None
            or validation_loader.batch_size != 32
            or loader.drop_last
            or validation_loader.drop_last
            or settings.get("microbatch_size") is not None
        ):
            raise ValueError(
                "Native ProDet retains whole training24 and validation32 batches,including final partial"
            )
        if runtime.device.type == "cuda":
            if settings.get("disk_saved_tensors", False):
                raise ValueError(
                    "CUDA ProDet uses native GPU autograd, not CPU audit disk transport"
                )
            self.validate_inputs(
                loader, validation_loader, settings.get("monitor", "video.auc")
            )
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise FileExistsError(
                    "ProDet fitting requires fresh output;preserve existing runs"
                )
            atomic_save(
                output / "source-data-protocol.json",
                {
                    "data_profile": self.data_profile,
                    "train": loader.dataset.selection_receipt(),
                    "validation": validation_loader.dataset.selection_receipt(),
                    "native_epochs": "zero-based0..20 inclusive21 by default",
                    "activation_transport": "native CUDA autograd;no retained CPU audit tensor files",
                    "source_only_selection": True,
                },
            )
            return super().fit(
                loader,
                validation_loader=validation_loader,
                runtime=runtime,
                settings={
                    "epochs": 21,
                    "learning_rate": 2e-4,
                    "weight_decay": 5e-4,
                    **settings,
                },
                output_dir=output,
            )
        if settings.get("disk_saved_tensors", True) is not True:
            raise ValueError(
                "ProDet CPU fitting requires exact disk saved-tensor transport"
            )
        self.validate_inputs(
            loader, validation_loader, settings.get("monitor", "video.auc")
        )
        epochs = integer(settings, "epochs", 21)
        # Candidate full-CNN meta graph estimates,not measured disk/RAM peaks.
        logical = (
            29262895872 if self.data_profile == "specialized_quadruples" else 7316105712
        )
        conservative_batch = math.ceil(logical * 1.25)
        required = conservative_batch * len(loader) * epochs
        self.activation_budget = (
            integer(
                settings, "saved_activation_budget_gib", math.ceil(required / 1024**3)
            )
            * 1024**3
        )
        self.disk_reserve = integer(settings, "disk_reserve_gib", 8) * 1024**3
        self.checkpoint_headroom, self.activation_bytes = 4 * 1024**3, 0
        if self.activation_budget < required:
            raise ValueError(
                "ProDet retained activation budget does not cover the declared whole-batch run"
            )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(
                "ProDet fitting requires fresh output;preserve existing runs"
            )
        protocol = {
            "data_profile": self.data_profile,
            "train": loader.dataset.selection_receipt(),
            "validation": validation_loader.dataset.selection_receipt(),
            "native_epochs": "zero-based0..20 inclusive21 by default",
            "logical_saved_bytes_per_full_batch": logical,
            "conservative_retained_run_bytes": required,
            "saved_activation_budget_bytes": self.activation_budget,
            "disk_reserve_bytes": self.disk_reserve,
            "checkpoint_headroom_bytes": self.checkpoint_headroom,
            "source_only_selection": True,
        }
        atomic_save(output / "source-data-protocol.json", protocol)
        free = shutil.disk_usage(output).free
        if free < self.activation_budget + self.disk_reserve + self.checkpoint_headroom:
            atomic_save(
                output / "resource-preflight.json",
                {"passed": False, "free_bytes": free, **protocol},
            )
            raise OSError(
                "ProDet full run cannot fit retained activations plus reserves;no training started"
            )
        atomic_save(
            output / "resource-preflight.json",
            {"passed": True, "free_bytes": free, **protocol},
        )
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 21,
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
