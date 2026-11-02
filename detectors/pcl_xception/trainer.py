"""Whole ordered I2G pairs, raw-mask PCL supervision and post-fit StepLR."""

from collections import Counter
import json
from pathlib import Path

import torch

from detectors._shared.paired import PairedImageTrainer
from detectors._shared.trainer import ImageClassificationTrainer
from detectors.fwa.generation import pixel_digest
from engine.evaluate import coverage
from engine.runtime import atomic_save
from .dataset import I2GDataset, I2GGenerationError, I2GPairCollator


class PCLTrainer(PairedImageTrainer):
    def __init__(self, model, *, origin=None, scheduler_state=None):
        super().__init__(model, origin=origin)
        self.scheduler, self.scheduler_state = None, scheduler_state
        self.actual_sources, self.replacements, self.component_sums = Counter(), 0, {}
        self.component_count = 0

    @staticmethod
    def validate_inputs(loader, validation_loader, monitor):
        ImageClassificationTrainer.validate_inputs(
            loader, validation_loader, monitor, training_labels={0}
        )
        # Foregrounds include every explicitly bound member, not just the real
        # background view. Check the complete inventory against selection data.
        train, val = loader.dataset.base.records, validation_loader.dataset.records

        def identity(r):
            return r.storage, r.path, r.row_group, r.row_index

        if {identity(r) for r in train} & {identity(r) for r in val} or (
            {r.video_id for r in train if r.video_id}
            & {r.video_id for r in val if r.video_id}
        ):
            raise ValueError("PCL complete foreground inventory overlaps validation")

    def validate_view_order(self, sources, metadata):
        expected = [
            (s["metadata"]["id"], kind)
            for kind in ("original", "generated")
            for s in sources
        ]
        if [(s["id"], s["view"]) for s in metadata] != expected:
            raise ValueError(
                "PCL requires all-real then all-fake order without shuffle"
            )

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        masks = batch.get("masks")
        if (
            not isinstance(masks, torch.Tensor)
            or masks.shape != (len(labels), 1, 256, 256)
            or masks.dtype != torch.float32
            or not torch.isfinite(masks).all()
        ):
            raise ValueError("PCL requires aligned finite raw float32 blend masks")
        sources = {s["metadata"]["id"]: s for s in batch["sources"]}
        for mask, view in zip(masks, batch["metadata"], strict=True):
            generation = sources[view["id"]]["generation"]
            if view["actual_background_id"] != generation["actual_background"]["id"]:
                raise ValueError(
                    "PCL requested view and actual background receipt differ"
                )
            if view["view"] == "original":
                if torch.count_nonzero(mask):
                    raise ValueError("PCL original view requires zero raw mask")
            elif pixel_digest(mask.numpy()) != generation["mask_sha256"]:
                raise ValueError("PCL generated raw mask differs from parent receipt")
        losses = self.model.loss_terms(
            self.model.forward_outputs(
                images, masks=masks.to(runtime.device, non_blocking=True)
            ),
            labels,
        )
        for key, value in losses.items():
            self.component_sums[key] = self.component_sums.get(key, 0.0) + float(
                value.detach()
            ) * len(labels)
        self.component_count += len(labels)
        return losses["overall"]

    def training_batches(self, loader, output):
        try:
            for batch in loader:
                for source in batch["sources"]:
                    generated = source["generation"]
                    self.actual_sources[generated["actual_background"]["id"]] += 1
                    self.replacements += int(generated["source_replacement"])
                yield batch
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            if receipt is None and I2GGenerationError.marker in str(error):
                try:
                    receipt = json.JSONDecoder().raw_decode(
                        str(error).split(I2GGenerationError.marker, 1)[1]
                    )[0]
                except (ValueError, TypeError):
                    pass
            atomic_save(
                output / "generation-failure.json",
                receipt or {"error": f"{type(error).__name__}: {error}"},
            )
            raise

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.actual_sources, self.replacements, self.component_sums = Counter(), 0, {}
        self.component_count = 0
        loss, report = super().train_epoch(
            loader, optimizer, scaler, runtime=runtime, output=output
        )
        actual = output / "actual-backgrounds"
        actual.mkdir()
        actual_report = coverage(loader.dataset, set(self.actual_sources), actual)
        report.update(
            coverage_unit="requested_background_slots",
            actual_background_coverage=actual_report,
            actual_background_occurrences=dict(self.actual_sources),
            source_decode_replacements=self.replacements,
        )
        atomic_save(output / "coverage.json", report)
        atomic_save(
            output / "loss-components.json",
            {
                "model_images": self.component_count,
                "mean": {
                    k: v / self.component_count for k, v in self.component_sums.items()
                },
                "objective": "CE+1*BCE(sigmoid_map,native4D_target); whole2B batch",
            },
        )
        return loss, report

    def build_optimizer(self, settings):
        if settings.get("optimizer", "adam") != "adam":
            raise ValueError("Configured native PCL requires Adam")
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
            raise ValueError("PCL snapshot requires explicit scheduler state")
        self.scheduler_state = state["scheduler"]
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if not isinstance(loader.dataset, I2GDataset) or not isinstance(
            loader.collate_fn, I2GPairCollator
        ):
            raise ValueError(
                "PCL requires bound real-only I2G dataset and ordered pair collator"
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
                "objective": "CE+1*BCE(sigmoid_map,raw-first-column4D)",
                "scheduler": "StepLR3_gamma0.4_once_after_complete_fit",
                "decode_failure": "one native first-source replacement; repeated first-source failure bounded/fatal",
                "identity": "Requested slot IDs unique; actual pixel background IDs separate; both coverage reports retained",
                "full_original_protocol_reproduced": False,
            },
        )
        result = super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 151,
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
