"""Whole-batch native freezing/BCE training with explicit method-state snapshots."""

from pathlib import Path

import torch

from detectors._shared.trainer import ImageClassificationTrainer, integer, number
from detectors.i3d.protocol import validate_source_splits
from detectors.i3d.trainer import I3DTrainer
from engine.runtime import atomic_save


class AltFreezingScorer:
    prediction_unit = "clip"

    def __init__(self, model):
        self.model = model

    def predict(self, clips, *, runtime):
        with runtime.autocast():
            return self.model(
                clips.to(runtime.device, non_blocking=True), inference=True
            ).view(-1)


class AltFreezingTrainer(I3DTrainer):
    def make_scorer(self):
        return AltFreezingScorer(self.model)

    def method_state_dict(self):
        return self.model.schedule_state_dict()

    def load_method_state_dict(self, state):
        self.model.load_schedule_state_dict(state)

    def build_optimizer(self, settings):
        # Native Adam groups are created before any freezing. Retain ALL weights,
        # including those currently frozen in a restored schedule, rather than
        # dropping them from the optimizer when they become trainable again.
        if settings.get("optimizer", "adam") != "adam":
            raise ValueError("The selected native AltFreezing recipe uses Adam")
        betas = settings.get("betas", [0.9, 0.999])
        if len(betas) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value < 1
            for value in betas
        ):
            raise ValueError("Adam betas must contain two values in[0,1)")
        return torch.optim.Adam(
            self.model.parameters(),
            lr=number(settings, "learning_rate", 0.0002, strict=True),
            weight_decay=number(settings, "weight_decay", 0.0005),
            betas=tuple(betas),
            eps=number(settings, "eps", 1e-8, strict=True),
        )

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if loader.batch_size != 8 or getattr(loader, "drop_last", False):
            raise ValueError(
                "Native AltFreezing uses whole batch8 and retains its final partial batch"
            )
        if runtime.precision != "fp32":
            raise ValueError("The selected AltFreezing alignment profile requires fp32")
        self.disk_activations = settings.get(
            "disk_saved_tensors", runtime.device.type == "cpu"
        )
        if type(self.disk_activations) is not bool or (
            self.disk_activations and runtime.device.type != "cpu"
        ):
            raise ValueError(
                "AltFreezing disk_saved_tensors is a CPU-only boolean policy"
            )
        self.activation_bytes = 0
        self.activation_budget = (
            integer(settings, "saved_activation_budget_gib", 20) * 1024**3
        )
        self.disk_reserve = integer(settings, "disk_reserve_gib", 8) * 1024**3
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(
                "AltFreezing fitting requires a fresh output; preserve old runs"
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
                "final_partial_batch": "retained",
                "freeze_schedule": "post-forward phase0 spatial,phase1 temporal,period21; only inferenceTrue suppresses it",
                "schedule_at_fit_start": self.method_state_dict(),
                "optimizer_parameter_membership": "all model parameters, even currently frozen weights",
                "disk_saved_tensors": self.disk_activations,
                "saved_activation_budget_bytes": self.activation_budget,
                "disk_reserve_bytes": self.disk_reserve,
                "saved_activations_retained": True,
                "full_experiment_reproduction": False,
            },
        )
        # Reuse I3D's complete-frame validation, probability BCE, whole-batch
        # saved-tensor context and the common selection/checkpoint loop. Its fit
        # precondition that every weight remains trainable is I3D-specific.
        return ImageClassificationTrainer.fit(
            self,
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 11,
                "monitor": "clip.auc",
                "learning_rate": 0.0002,
                "weight_decay": 0.0005,
                **settings,
            },
            output_dir=output,
        )
