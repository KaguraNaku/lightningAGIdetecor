"""Method-owned AdamW, batch-coupled loss and step-wise cyclic scheduling."""
from pathlib import Path
import torch
from detectors._shared.trainer import ImageClassificationTrainer
from .loss import GenDLoss


class GenDTrainer(ImageClassificationTrainer):
    supports_gradient_cache = True

    def __init__(self, model, *, origin=None):
        super().__init__(model, origin=origin, objective=GenDLoss())
        self.steps = 0
        self.scheduler = None
        self.schedule_spec = None
        self.parent_schedule = None
        self.gradient_cache = None
        self.gradient_cache_policy = None

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        return self.objective(self.model.forward_outputs(images), labels)

    def configure_gradient_cache(self, settings, logical_batch_size):
        """Keep loader batches and optimizer/scheduler steps unchanged."""
        size = settings.get("gradient_cache_microbatch_size")
        if size is None:
            self.gradient_cache = self.gradient_cache_policy = None
            return
        from .gradient_cache import GenDGradientCache
        self.gradient_cache = GenDGradientCache(self.model, size)
        if size > logical_batch_size:
            raise ValueError("gradient_cache_microbatch_size cannot exceed the logical loader batch size")
        self.gradient_cache_policy = self.gradient_cache.policy(logical_batch_size)

    def backward_batch(self, batch, images, labels, *, runtime, scaler):
        if self.gradient_cache is None:
            return super().backward_batch(batch, images, labels, runtime=runtime, scaler=scaler)
        try:
            return self.gradient_cache.backward(images, labels, objective=self.objective,
                scaler=scaler, runtime=runtime, validate_loss=self.validate_loss)
        finally:
            timing = getattr(self, "_timing", None)
            if timing is not None and timing.meter is not None:
                cache = self.gradient_cache
                timing.meter.metadata.update(gradient_cache=self.gradient_cache_policy,
                    effective_batch_size=self.gradient_cache_policy["logical_batch_size"],
                    execution_microbatch_size=cache.microbatch_size,
                    gradient_cache_gpu_time_scope="cache forward+full logical loss/output gradient+replay forward/backward;optimizer included by train_step region",
                    cache_forward_images=cache.cached_images,
                    replay_forward_images=cache.replayed_images,
                    model_forward_images=cache.cached_images + cache.replayed_images,
                    cache_forward_batches=cache.cache_forwards,
                    replay_forward_batches=cache.replay_forwards)

    def fit(self, loader, **kwargs):
        settings = kwargs["settings"]
        self.configure_gradient_cache(settings, loader.batch_size)
        if self.gradient_cache is not None and kwargs["runtime"].precision != "fp32":
            raise ValueError("GenD gradient caching requires the native FP32 protocol")
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
            if method.get("gradient_cache") != self.gradient_cache_policy:
                raise ValueError("GenD gradient-cache execution policy changed at a stage boundary")
            self.parent_schedule = method["schedule_spec"]
            del parent
        report = super().fit(loader, **kwargs)
        if self.gradient_cache is not None:
            report["gradient_cache"] = self.gradient_cache_policy
        return report

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
                "schedule_spec": self.schedule_spec, "scheduler": self.scheduler.state_dict(),
                "gradient_cache": self.gradient_cache_policy}

    def load_method_state_dict(self, state):
        if not state or state.get("profile") != "gend-author-ln-l2-ua" or state["schedule_spec"] != self.schedule_spec:
            raise ValueError("Mismatched GenD objective/scheduler checkpoint")
        if state.get("gradient_cache") != self.gradient_cache_policy:
            raise ValueError("Mismatched GenD gradient-cache execution policy")
        self.steps = state["steps"]
        self.scheduler.load_state_dict(state["scheduler"])
