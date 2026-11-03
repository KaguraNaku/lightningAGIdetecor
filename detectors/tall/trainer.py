"""Whole native T4 CE/Adam with explicit source-validation and data roles."""

from contextlib import contextmanager, nullcontext
from pathlib import Path
import shutil

from torch.multiprocessing.reductions import StorageWeakRef

from detectors._shared.trainer import ImageClassificationTrainer, integer
from engine.runtime import atomic_save
from engine.saved_tensors import DiskSavedTensors
from .dataset import TALLDataset
from .protocol import validate_source_splits


class TALLTrainer(ImageClassificationTrainer):
    prediction_unit = "clip"

    def build_optimizer(self, settings):
        if settings.get("optimizer", "adam") != "adam":
            raise ValueError("The selected native TALL profile uses Adam")
        if any(not p.requires_grad for p in self.model.parameters()):
            raise ValueError("Native TALL trains every model parameter")
        return super().build_optimizer({"learning_rate": 2e-5, **settings})

    def compute_loss(self, clips, labels):
        outputs = self.model.forward_outputs(clips)
        return self.model.loss_terms(outputs, labels)["overall"]

    @staticmethod
    def validate_inputs(loader, validation_loader, monitor, *, training_labels=None):
        if validation_loader is None:
            raise ValueError("TALL requires source-domain fit.validation_input")
        if not isinstance(loader.dataset, TALLDataset) or not isinstance(
            validation_loader.dataset, TALLDataset
        ):
            raise ValueError("TALL requires explicit complete-frame clip adapters")
        if (
            loader.dataset.stage != "train"
            or validation_loader.dataset.stage != "validation"
        ):
            raise ValueError(
                "TALL train/validation adapters have different stage roles"
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
            storage = tensor.untyped_storage()
            identity = StorageWeakRef(storage)
            key = identity.cdata, tensor._version, str(tensor.dtype)
            needs_write = (
                storage.data_ptr() not in store.persistent
                and storage.nbytes() >= store.threshold
                and key not in store.cache
            )
            if needs_write:
                possible_write = storage.nbytes() + 256 * 1024
                written = sum(row["bytes"] for row in store.files)
                if before + written + possible_write > self.activation_budget:
                    raise OSError(
                        "TALL preserved activation budget would be exceeded; no data removed"
                    )
                reserve = self.disk_reserve + getattr(self, "checkpoint_headroom", 0)
                if shutil.disk_usage(output).free - possible_write < reserve:
                    raise OSError("TALL saved activations would cross the disk reserve")
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
                retention="Preserve saved activations on success and failure;no cleanup",
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
            loader.batch_size != 32
            or validation_loader is None
            or validation_loader.batch_size != 32
            or getattr(loader, "drop_last", False)
            or getattr(validation_loader, "drop_last", False)
        ):
            raise ValueError(
                "Native TALL uses batch_size32 and retains its final partial batch"
            )
        if runtime.precision != "fp32":
            raise ValueError("The selected native TALL alignment recipe requires fp32")
        if any(not p.requires_grad for p in self.model.parameters()):
            raise ValueError("The native TALL recipe trains all model parameters")
        self.validate_inputs(
            loader, validation_loader, settings.get("monitor", "clip.auc")
        )
        self.disk_activations = settings.get(
            "disk_saved_tensors", runtime.device.type == "cpu"
        )
        if type(self.disk_activations) is not bool or (
            self.disk_activations and runtime.device.type != "cpu"
        ):
            raise ValueError("TALL disk_saved_tensors is a CPU-only boolean policy")
        self.activation_bytes = 0
        self.activation_budget = (
            integer(settings, "saved_activation_budget_gib", 20) * 1024**3
        )
        self.disk_reserve = integer(settings, "disk_reserve_gib", 8) * 1024**3
        # The fixed full model has347.7MB parameters. Three full Adam/model
        # archives can coexist during atomic last/best replacement;reserve3GiB
        # beyond the free-space floor so activation writes cannot consume it.
        self.checkpoint_headroom = 3 * 1024**3
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(
                "TALL fitting requires a fresh output directory;preserve old runs"
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
                "native_batch_size": 32,
                "final_partial_batch": "retained,as in original train loader",
                "data_profile": loader.dataset.profile,
                "validation_stochastic": loader.dataset.profile
                in {"native_entrypoint", "specialized_cutout"},
                "native_process_RNG_reproduced": False,
                "disk_saved_tensors": self.disk_activations,
                "saved_activation_budget_bytes": self.activation_budget,
                "disk_reserve_bytes": self.disk_reserve,
                "checkpoint_write_headroom_bytes": self.checkpoint_headroom,
                "saved_activations_retained": True,
                "full_experiment_reproduction": False,
            },
        )
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 101,
                "monitor": "clip.auc",
                "learning_rate": 2e-5,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
