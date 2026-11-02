"""Native CE/Gram training and once-after-complete-fit cosine scheduling."""

from pathlib import Path
import torch

from detectors._shared.trainer import ImageClassificationTrainer, integer
from engine.runtime import atomic_save, release_cpu_memory
from .scheduler import SourceWarmupScheduler
from .checkpoint import restore_weights


class GOANetTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None, scheduler_state=None):
        super().__init__(model, origin=origin)
        self.scheduler = None
        self.scheduler_state = scheduler_state
        self.warmup_epoch = 1

    def load_selected_weights(self, weights):
        restore_weights(self.model, weights)
        release_cpu_memory()

    def compute_loss(self, images, labels):
        return self.model.loss_terms(self.model.forward_outputs(images), labels)[
            "overall"
        ]

    def build_optimizer(self, settings):
        optimizer = super().build_optimizer(
            {"learning_rate": 1e-4, "weight_decay": 5e-4, **settings}
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5
        )
        default_warmup = (
            self.scheduler_state["warmup_epoch"]
            if self.scheduler_state is not None
            else 1
        )
        self.warmup_epoch = integer(settings, "warmup_epoch", default_warmup, minimum=0)
        if self.warmup_epoch:
            self.scheduler = SourceWarmupScheduler(
                optimizer, self.warmup_epoch, self.scheduler
            )
        if self.scheduler_state is not None:
            if self.scheduler_state["warmup_epoch"] != self.warmup_epoch:
                raise ValueError(
                    "GOANet saved scheduler warmup differs from fit settings"
                )
            self.scheduler.load_state_dict(self.scheduler_state["scheduler"])
            for group, rate in zip(
                optimizer.param_groups, self.scheduler.get_last_lr(), strict=True
            ):
                group["lr"] = rate
        return optimizer

    def method_state_dict(self):
        return {
            "scheduler": {
                "warmup_epoch": self.warmup_epoch,
                "scheduler": self.scheduler.state_dict(),
            }
            if self.scheduler is not None
            else self.scheduler_state
        }

    def load_method_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"scheduler"}:
            raise ValueError("GOANet trainer requires explicit scheduler state")
        self.scheduler_state = state["scheduler"]
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state["scheduler"])

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        report = super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={"epochs": 11, "learning_rate": 1e-4, **settings},
            output_dir=output_dir,
        )
        # The checked non-SWA source steps once after the entire epoch loop.
        self.scheduler.step()
        self.scheduler_state = {
            "warmup_epoch": self.warmup_epoch,
            "scheduler": self.scheduler.state_dict(),
        }
        atomic_save(
            Path(output_dir) / "source-scheduler.json",
            {
                "policy": "once_after_complete_fit",
                "state": self.scheduler_state,
                "learning_rates": self.scheduler.get_last_lr(),
            },
        )
        self.scheduler = None
        report["scheduler"] = "source-scheduler.json"
        return report
