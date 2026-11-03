"""Native whole-T8 two-logit CE within the shared source-validation loop."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import shutil

import torch

from detectors._shared.trainer import ImageClassificationTrainer, integer
from engine.runtime import atomic_save
from engine.saved_tensors import DiskSavedTensors
from .dataset import STILDataset
from .protocol import validate_source_splits


class STILTrainer(ImageClassificationTrainer):
    prediction_unit = "clip"

    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin, objective=torch.nn.CrossEntropyLoss())

    def compute_loss(self, clips, labels):
        return self.model.loss_terms(self.model.forward_outputs(clips), labels)[
            "overall"
        ]

    @staticmethod
    def validate_inputs(loader, validation_loader, monitor, *, training_labels=None):
        if validation_loader is None:
            raise ValueError("STIL requires source-domain fit.validation_input")
        if not isinstance(loader.dataset, STILDataset) or not isinstance(
            validation_loader.dataset, STILDataset
        ):
            raise ValueError("STIL needs its explicit complete-frame clip adapters")
        if (
            loader.dataset.stage != "train"
            or validation_loader.dataset.stage != "validation"
        ):
            raise ValueError(
                "STIL train/validation adapters have different stage roles"
            )
        validate_source_splits(loader.dataset, validation_loader.dataset)

    @contextmanager
    def _disk_batch(self, *, output, step):
        store = DiskSavedTensors(
            self.model, Path(output) / f"batch-{step:06d}-saved-tensors"
        )
        pack = store.pack
        before = self.activation_bytes

        def budgeted_pack(tensor):
            possible_write = tensor.untyped_storage().nbytes() + 256 * 1024
            written = sum(row["bytes"] for row in store.files)
            if (
                self.activation_bytes + written + possible_write
                > self.activation_budget
            ):
                raise OSError(
                    "STIL preserved activation budget would be exceeded; no data removed"
                )
            if shutil.disk_usage(output).free - possible_write < self.disk_reserve:
                raise OSError("STIL saved activations would cross the disk reserve")
            return pack(tensor)

        store.pack = budgeted_pack
        try:
            with store:
                yield
        finally:
            receipt = store.report()
            self.activation_bytes = before + receipt["bytes_written"]
            receipt.update(
                activation_bytes_this_fit=self.activation_bytes,
                activation_budget_bytes=self.activation_budget,
                retention="Preserve this run's saved activations on success and failure; no cleanup",
            )
            atomic_save(Path(output) / f"batch-{step:06d}-saved-tensors.json", receipt)

    def batch_context(self, batch, *, runtime, output, step):
        return (
            self._disk_batch(output=output, step=step)
            if self.disk_activations
            else nullcontext()
        )

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if (
            loader.batch_size != 8
            or validation_loader is None
            or validation_loader.batch_size != 8
            or getattr(loader, "drop_last", False)
            or getattr(validation_loader, "drop_last", False)
        ):
            raise ValueError(
                "Native STIL uses batch_size8 and retains its final partial batch"
            )
        if runtime.precision != "fp32":
            raise ValueError("The selected native STIL alignment recipe requires fp32")
        if any(not parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("The native STIL recipe trains all model parameters")
        self.disk_activations = settings.get(
            "disk_saved_tensors", runtime.device.type == "cpu"
        )
        if type(self.disk_activations) is not bool or (
            self.disk_activations and runtime.device.type != "cpu"
        ):
            raise ValueError("STIL disk_saved_tensors is a CPU-only boolean policy")
        self.activation_bytes = 0
        self.activation_budget = (
            integer(settings, "saved_activation_budget_gib", 20) * 1024**3
        )
        self.disk_reserve = integer(settings, "disk_reserve_gib", 8) * 1024**3
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(
                "STIL fitting requires a fresh output directory; preserve old runs"
            )
        self.validate_inputs(
            loader, validation_loader, settings.get("monitor", "clip.auc")
        )
        atomic_save(
            output / "source-pool-protocol.json",
            {
                "train": loader.dataset.protocol,
                "validation": validation_loader.dataset.protocol,
                "source_isolation": validate_source_splits(
                    loader.dataset, validation_loader.dataset
                ),
                "prediction_unit": "clip",
                "native_batch_size": 8,
                "final_partial_batch": "retained, as in original train loader",
                "disk_saved_tensors": self.disk_activations,
                "saved_activation_budget_bytes": self.activation_budget,
                "disk_reserve_bytes": self.disk_reserve,
                "saved_activations_retained": True,
                "full_experiment_reproduction": False,
            },
        )
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 11,
                "monitor": "clip.auc",
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
