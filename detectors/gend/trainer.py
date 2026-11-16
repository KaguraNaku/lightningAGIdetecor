"""Method-owned AdamW, batch-coupled loss and step-wise cyclic scheduling."""
from pathlib import Path
import torch
from detectors._shared.trainer import ImageClassificationTrainer
from .loss import GenDLoss


class GenDTrainer(ImageClassificationTrainer):
    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin, objective=GenDLoss())
        self.steps = 0
        self.scheduler = None
        self.schedule_spec = None
        self.parent_schedule = None

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        return self.objective(self.model.forward_outputs(images), labels)

    def fit(self, loader, **kwargs):
        settings = kwargs["settings"]
        self.steps_per_epoch = len(loader)
        if self.steps_per_epoch < 1 or len(loader.dataset) % loader.batch_size == 1:
            raise ValueError("GenD cannot silently drop or train a singleton uniformity batch")
        self.parent_schedule = None
        stage = settings.get("continuation", {})
        if stage.get("stage_index", 0) > 0:
            parent = torch.load(Path(stage["previous_best"]["path"]), map_location="cpu", weights_only=True, mmap=True)
            method = parent.get("method_state") or {}
            if method.get("profile") != "gend-author-ln-l2-ua":
                raise ValueError("Cannot continue a historical CE-only chain as full GenD")
            self.parent_schedule = method["schedule_spec"]
            del parent
        return super().fit(loader, **kwargs)

    def build_optimizer(self, settings):
        if settings.get("optimizer", "adamw") != "adamw" or settings.get("weight_decay", 0.) != 0.:
            raise ValueError("GenD author profile requires AdamW with zero weight decay")
        rate = float(settings.get("learning_rate", 3e-4))
        low = float(settings.get("min_learning_rate", 1e-6))
        cycle = int(settings.get("cycle_epochs", 10))
        warmup = int(settings.get("warmup_epochs", 0))
        if not 0 < low < rate or cycle < 1 or warmup < 0:
            raise ValueError("Invalid GenD cyclic learning-rate schedule")
        self.schedule_spec = {"learning_rate": rate, "minimum": low, "cycle_epochs": cycle,
                              "warmup_epochs": warmup, "period_steps": cycle * self.steps_per_epoch,
                              "warmup_steps": warmup * self.steps_per_epoch}
        if self.parent_schedule is not None:
            for key in ("learning_rate", "minimum", "cycle_epochs", "warmup_epochs"):
                if self.schedule_spec[key] != self.parent_schedule[key]:
                    raise ValueError("GenD scheduler policy changed at a stage boundary")
            self.schedule_spec = self.parent_schedule.copy()
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                     lr=rate, weight_decay=0., betas=(.9, .999), eps=1e-8)
        cyclic = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.schedule_spec["period_steps"], T_mult=1, eta_min=low)
        if self.schedule_spec["warmup_steps"]:
            ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=low/rate,
                                                    total_iters=self.schedule_spec["warmup_steps"])
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [ramp, cyclic], milestones=[self.schedule_spec["warmup_steps"]])
        else:
            self.scheduler = cyclic
        optimizer.register_step_post_hook(self._after_step)
        return optimizer

    def _after_step(self, optimizer, args, kwargs):
        self.steps += 1
        self.scheduler.step()

    def validate_optimizer_contract(self, before, carried):
        # The selected step's LR is dynamic; the immutable base rates/schedule
        # are separately checked in load_method_state_dict, then fully restored.
        strip = lambda groups: [{k:v for k,v in g.items() if k not in ("lr", "initial_lr")} for g in groups]
        if strip(before) != strip(carried):
            raise ValueError("GenD optimizer hyperparameters changed")

    def method_state_dict(self):
        return {"version": 1, "profile": "gend-author-ln-l2-ua", "steps": self.steps,
                "schedule_spec": self.schedule_spec, "scheduler": self.scheduler.state_dict()}

    def load_method_state_dict(self, state):
        if not state or state.get("profile") != "gend-author-ln-l2-ua" or state["schedule_spec"] != self.schedule_spec:
            raise ValueError("Mismatched GenD objective/scheduler checkpoint")
        self.steps = state["steps"]
        self.scheduler.load_state_dict(state["scheduler"])
