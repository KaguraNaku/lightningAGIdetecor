"""Whole grouped LSDA training; sampled inventory coverage stays method-owned."""

from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from detectors._shared.trainer import ImageClassificationTrainer
from engine.runtime import atomic_save, release_cpu_memory
from engine.saved_tensors import DiskSavedTensors
from .dataset import LSDADataset, LSDACollator, metadata
from .sampler import LSDAGroupSampler


class LSDATrainer(ImageClassificationTrainer):
    def build_optimizer(self, settings):
        optimizer = super().build_optimizer(settings)
        # Keep shared hyperparameter validation, but native LSDA puts ALL
        # parameters in one group, including frozen and unused parameters.
        optimizer.param_groups[0]["params"] = list(self.model.parameters())
        return optimizer

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.model.train()
        sources = {row.id: row for row in loader.dataset.records}
        groups, excluded, requested_ids, actual_ids = set(), set(), Counter(), Counter()
        components, images_seen, loss_sum, batches = Counter(), 0, 0.0, 0
        with (output / "sampled-sources.jsonl").open("w") as stream:
            for step, batch in enumerate(loader):
                specific = batch["specific_labels"]
                count = specific.numel()
                if (
                    specific.ndim != 2
                    or specific.shape[1] != 5
                    or batch["pixel_values"].shape
                    != (specific.shape[0], 5, 3, 256, 256)
                    or specific.tolist() != [list(range(5))] * specific.shape[0]
                    or batch["labels"].tolist() != [[0, 1, 1, 1, 1]] * specific.shape[0]
                    or len(batch["metadata"]) != count
                    or len(batch["receipts"]) != count
                ):
                    raise ValueError(
                        "LSDA requires aligned canonical five-domain grouped batches"
                    )
                for index in range(specific.shape[0]):
                    receipts = batch["receipts"][index * 5 : (index + 1) * 5]
                    requested = [
                        sources.get(row["requested"]["id"]) for row in receipts
                    ]
                    if any(row is None for row in requested):
                        raise ValueError(
                            "LSDA sampled a requested source outside its inventory"
                        )
                    group = requested[0].metadata["lsda_group"]
                    slot = requested[0].metadata["lsda_frame_slot"]
                    if group in groups or group in excluded:
                        raise ValueError(
                            "LSDA must sample each group exactly once per epoch"
                        )
                    for domain, (source, receipt) in enumerate(
                        zip(requested, receipts, strict=True)
                    ):
                        actual = sources.get(receipt["actual"]["id"])
                        observed = batch["metadata"][index * 5 + domain]
                        if (
                            source.metadata["lsda_group"] != group
                            or source.metadata["lsda_frame_slot"] != slot
                            or source.metadata["specific_label"] != domain
                            or metadata(source) != receipt["requested"]
                            or actual is None
                            or metadata(actual) != receipt["actual"]
                            or observed != receipt["actual"]
                            or actual.metadata["lsda_group"] != group
                            or actual.metadata["specific_label"] != domain
                            or actual.video_id != source.video_id
                        ):
                            raise ValueError(
                                "LSDA requested/actual source transport changed group, video or domain"
                            )
                        requested_ids[source.id] += 1
                        actual_ids[actual.id] += 1
                    groups.add(group)
                for rejected in batch["rejected_groups"]:
                    bindings = {
                        row["attributes"]["lsda_group"] for row in rejected["sources"]
                    }
                    if len(bindings) != 1:
                        raise ValueError("Rejected LSDA group mixes source groups")
                    group = next(iter(bindings))
                    if group in groups or group in excluded:
                        raise ValueError(
                            "LSDA repeated an accepted or rejected source group"
                        )
                    excluded.add(group)
                stream.write(
                    json.dumps(
                        {
                            "batch": step,
                            "accepted": batch["receipts"],
                            "rejected": batch["rejected_groups"],
                        }
                    )
                    + "\n"
                )
                stream.flush()
                optimizer.zero_grad(set_to_none=True)
                images = batch["pixel_values"].to(runtime.device, non_blocking=True)
                labels = specific.to(runtime.device, non_blocking=True)
                store = (
                    DiskSavedTensors(
                        self.model, output / f"batch-{step:06d}-saved-tensors"
                    )
                    if self.disk_activations
                    else None
                )
                try:
                    with (
                        runtime.autocast(),
                        store if store is not None else nullcontext(),
                    ):
                        predicted = self.model.forward_outputs(images, labels)
                        losses = self.model.loss_terms(predicted)
                        loss = losses["overall"]
                        if loss.ndim != 0 or not all(
                            torch.isfinite(value).all() for value in losses.values()
                        ):
                            raise ValueError(
                                "LSDA native objectives must be finite scalars"
                            )
                        scaler.scale(loss).backward()
                    for name, value in losses.items():
                        components[name] += float(value.detach()) * count
                    loss_sum += float(loss.detach()) * count
                    gradient_count = sum(
                        parameter.grad is not None
                        for parameter in self.model.parameters()
                    )
                    del predicted, losses, loss
                    if store is not None:
                        store.cleanup()
                    release_cpu_memory()
                    scaler.step(optimizer)
                    scaler.update()
                finally:
                    if store is not None:
                        atomic_save(
                            output / f"batch-{step:06d}-transport.json", store.report()
                        )
                images_seen += count
                batches += 1
                atomic_save(
                    output / f"batch-{step:06d}-update.json",
                    {
                        "images": count,
                        "groups": specific.shape[0],
                        "gradient_tensors": gradient_count,
                        "optimizer_parameter_tensors": sum(
                            len(group["params"]) for group in optimizer.param_groups
                        ),
                    },
                )
        if (
            groups & excluded
            or groups | excluded != set(range(loader.dataset.num_groups))
            or not images_seen
        ):
            raise ValueError("LSDA sampled-group coverage is incomplete")
        with (output / "unselected-inventory.jsonl").open("w") as stream:
            for row in loader.dataset.records:
                if row.id not in requested_ids:
                    stream.write(
                        json.dumps(
                            {
                                "id": row.id,
                                "reason": "invalid_group_excluded"
                                if row.metadata["lsda_group"] in excluded
                                else "native_one_slot_per_group_sampling",
                            }
                        )
                        + "\n"
                    )
        report = {
            "inventory_rows": len(loader.dataset),
            "expected_groups": loader.dataset.num_groups,
            "successful_groups": len(groups),
            "excluded_groups": sorted(excluded),
            "sampled_images": images_seen,
            "batches": batches,
            "requested_counts": dict(requested_ids),
            "actual_counts": dict(actual_ids),
            "unselected_inventory_rows": len(loader.dataset) - len(requested_ids),
            "selection_policy": "Native one shared slot per five-domain group, not all32 inventory slots each epoch",
        }
        atomic_save(output / "coverage.json", report)
        atomic_save(
            output / "losses.json",
            {
                "images": images_seen,
                "mean": {key: value / images_seen for key, value in components.items()},
            },
        )
        return loss_sum / images_seen, report

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if (
            not isinstance(loader.dataset, LSDADataset)
            or not isinstance(loader.collate_fn, LSDACollator)
            or not isinstance(loader.sampler, LSDAGroupSampler)
        ):
            raise ValueError(
                "LSDA training requires grouped inventory, collator and native sampler"
            )
        if (
            loader.batch_size != loader.sampler.batch_size
            or len(loader.sampler) != loader.dataset.num_groups * 5
        ):
            raise ValueError(
                "LSDA sampler batch/inventory geometry differs from its loader"
            )
        self.disk_activations = settings.get(
            "disk_saved_tensors", runtime.device.type == "cpu"
        )
        if type(self.disk_activations) is not bool:
            raise ValueError("LSDA disk_saved_tensors must be boolean")
        if self.disk_activations and runtime.device.type != "cpu":
            raise ValueError("LSDA disk saved-tensor transport is CPU-only")
        output = Path(output_dir)
        atomic_save(output / "group-transport.json", loader.dataset.selection_receipt())
        return super().fit(
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
