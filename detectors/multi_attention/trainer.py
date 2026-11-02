"""Two-stage training with explicit inference routing and native state snapshots."""

from pathlib import Path

import torch

from detectors._shared.trainer import ImageClassificationTrainer
from engine.runtime import atomic_save

from .state import pack_state, restore_state


class InferenceScorer:
    def __init__(self, model):
        self.model = model

    def predict(self, images, *, runtime):
        with runtime.autocast():
            return self.model(
                images.to(runtime.device, non_blocking=True), inference=True
            ).softmax(1)[:, 1]


class MultiAttentionTrainer(ImageClassificationTrainer):
    def __init__(self, model, objective, *, origin=None, scheduler_state=None):
        super().__init__(model, origin=origin, objective=objective)
        self.scheduler = None
        self.scheduler_state = scheduler_state

    def make_scorer(self):
        return InferenceScorer(self.model)

    def compute_loss(self, images, labels):
        outputs = self.model.forward_outputs(images, inference=False)
        return self.objective(
            outputs, labels, late=self.model.batch_cnt > self.model.warmup_batches
        )

    def build_optimizer(self, settings):
        optimizer = super().build_optimizer({"weight_decay": 1e-6, **settings})
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        return optimizer

    def method_state_dict(self):
        scheduler = (
            self.scheduler.state_dict()
            if self.scheduler is not None
            else self.scheduler_state
        )
        return pack_state(self.model, self.objective, scheduler)

    def load_method_state_dict(self, state):
        self.scheduler_state = restore_state(self.model, self.objective, state)
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        output_dir = Path(output_dir)
        report = super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={"epochs": 51, "weight_decay": 1e-6, **settings},
            output_dir=output_dir,
        )
        # The checked local train.py calls this once AFTER its complete epoch
        # loop. Preserve that placement rather than silently repairing it.
        self.scheduler.step()
        self.scheduler_state = self.scheduler.state_dict()
        atomic_save(
            output_dir / "source-scheduler.json",
            {
                "policy": "once_after_complete_fit",
                "state": self.scheduler_state,
                "learning_rates": self.scheduler.get_last_lr(),
            },
        )
        self.scheduler = None  # Release the optimizer held by StepLR.
        report["scheduler"] = "source-scheduler.json"
        return report
