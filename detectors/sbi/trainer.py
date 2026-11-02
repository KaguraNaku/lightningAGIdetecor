"""Ordered real-first SBI pairs, three-logit CE and native post-fit StepLR."""

import json
from pathlib import Path

import torch

from detectors._shared.paired import PairedImageTrainer
from detectors._shared.trainer import ImageClassificationTrainer
from engine.runtime import atomic_save
from .dataset import SBIRealDataset
from .generation import SBIPairCollator, SBIGenerationError


class ThreeLogitScorer:
    def __init__(self, model):
        self.model = model

    def predict(self, images, *, runtime):
        with runtime.autocast():
            return self.model(images.to(runtime.device, non_blocking=True)).softmax(1)[
                :, 1
            ]


class SBITrainer(PairedImageTrainer):
    fallback_statistic = "generation_fallbacks"

    def __init__(self, model, *, origin=None, scheduler_state=None):
        super().__init__(model, origin=origin)
        self.scheduler = None
        self.scheduler_state = scheduler_state

    @staticmethod
    def validate_inputs(loader, validation_loader, monitor):
        return ImageClassificationTrainer.validate_inputs(
            loader, validation_loader, monitor, training_labels={0}
        )

    def compute_loss(self, images, labels):
        logits = self.model(images)
        if logits.shape != (len(labels), 3):
            raise ValueError("Configured SBI requires all three logits")
        return self.objective(logits, labels)

    def make_scorer(self):
        return ThreeLogitScorer(self.model)

    def validate_view_order(self, sources, metadata):
        expected = [
            (item["metadata"]["id"], kind)
            for kind in ("original", "generated")
            for item in sources
        ]
        if [(item["id"], item["view"]) for item in metadata] != expected:
            raise ValueError(
                "SBI batches must retain all-real then all-fake source order without shuffle"
            )

    def training_batches(self, loader, output):
        try:
            yield from loader
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            marker = SBIGenerationError.marker
            if receipt is None and marker in str(error):
                text = str(error).split(marker, 1)[1]
                try:
                    receipt = json.JSONDecoder().raw_decode(text)[0]
                except (ValueError, TypeError):
                    receipt = {"error": str(error)}
            atomic_save(
                output / "generation-failure.json",
                receipt or {"error": f"{type(error).__name__}: {error}"},
            )
            raise

    def build_optimizer(self, settings):
        optimizer = super().build_optimizer(settings)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.4
        )
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
            for group, rate in zip(
                optimizer.param_groups, self.scheduler.get_last_lr(), strict=True
            ):
                group["lr"] = rate
        return optimizer

    def method_state_dict(self):
        return {
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else self.scheduler_state
        }

    def load_method_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"scheduler"}:
            raise ValueError("SBI snapshot requires explicit scheduler state")
        self.scheduler_state = state["scheduler"]
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if not isinstance(loader.dataset, SBIRealDataset) or not isinstance(
            loader.collate_fn, SBIPairCollator
        ):
            raise ValueError(
                "SBI requires a real-only dataset view and native ordered pair collator"
            )
        output = Path(output_dir)
        atomic_save(
            output / "source-selection.json", loader.dataset.selection_receipt()
        )
        atomic_save(
            output / "paired-protocol.json",
            {
                "source_batch_size": loader.batch_size,
                "model_batch_size": 2 * loader.batch_size,
                "order": "all_real_then_all_fake_no_shuffle",
                "logits": 3,
                "fake_score": "softmax_all3_class1",
                "scheduler": "StepLR3_gamma0.4_once_after_complete_fit",
                "generation_failures": "Fatal with original cause and RNG retained; no relabel/black-image repair",
            },
        )
        result = super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 51,
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
        self.scheduler.step()
        self.scheduler_state = self.scheduler.state_dict()
        atomic_save(
            output / "source-scheduler.json",
            {
                "policy": "once_after_complete_fit",
                "state": self.scheduler_state,
                "learning_rates": self.scheduler.get_last_lr(),
            },
        )
        self.scheduler = None
        result["scheduler"] = "source-scheduler.json"
        return result
