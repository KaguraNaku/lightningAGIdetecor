"""Identity-supervised whole batches; repeated replacement sources remain explicit."""

from collections import Counter
import json
from pathlib import Path

import torch

from detectors._shared.trainer import ImageClassificationTrainer
from engine.evaluate import coverage
from engine.runtime import atomic_save
from .dataset import IIDDataset, IIDCollator, IIDSampleError


class IIDTrainer(ImageClassificationTrainer):
    def build_optimizer(self, settings):
        optimizer = super().build_optimizer(settings)
        # Native Adam includes the frozen final identity-BN weight in its ordered
        # parameter group; its absent gradient prevents updates/weight decay.
        optimizer.param_groups[0]["params"] = list(self.model.parameters())
        return optimizer

    def training_batches(self, loader, output):
        try:
            yield from loader
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            if receipt is None and IIDSampleError.marker in str(error):
                try:
                    receipt = json.JSONDecoder().raw_decode(
                        str(error).split(IIDSampleError.marker, 1)[1]
                    )[0]
                except (ValueError, TypeError):
                    pass
            atomic_save(
                output / "source-failure.json",
                receipt or {"error": f"{type(error).__name__}: {error}"},
            )
            raise

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.model.train()  # Includes the native trainable explicit extractor.
        expected = {r.id: r for r in loader.dataset.records}
        requested, actual, components = set(), Counter(), Counter()
        loss_sum, images_count, replacements = 0.0, 0, 0
        with (output / "samples.partial.jsonl").open("w") as stream:
            for step, batch in enumerate(self.training_batches(loader, output), 1):
                count = len(batch["labels"])
                if (
                    len(batch["pixel_values"]) != count
                    or len(batch["metadata"]) != count
                    or len(batch["requests"]) != count
                    or batch["id_indices"].shape != (count,)
                ):
                    raise ValueError(
                        "IID batch image/label/identity/request alignment mismatch"
                    )
                for request, metadata, label, identity in zip(
                    batch["requests"],
                    batch["metadata"],
                    batch["labels"].tolist(),
                    batch["id_indices"].tolist(),
                    strict=True,
                ):
                    wanted, observed = (
                        expected.get(request["metadata"]["id"]),
                        expected.get(metadata["id"]),
                    )
                    if (
                        wanted is None
                        or wanted.id in requested
                        or observed is None
                        or request["label"] != wanted.label
                        or label != observed.label
                        or identity != observed.metadata["identity_index"]
                    ):
                        raise ValueError(
                            "IID invalid/repeated requested slot or misaligned actual identity/label"
                        )
                    for values, row in (
                        (request["metadata"], wanted),
                        (metadata, observed),
                    ):
                        if (
                            values["dataset"] != row.dataset
                            or values["split"] != row.split
                            or values["video_id"] != row.video_id
                            or values["identity_index"]
                            != row.metadata["identity_index"]
                        ):
                            raise ValueError(
                                "IID source metadata differs from manifest identity"
                            )
                    if not request["replacement"] and wanted.id != observed.id:
                        raise ValueError("IID unexpected source replacement")
                    if (
                        request["replacement"]
                        and observed.id != loader.dataset.records[0].id
                    ):
                        raise ValueError(
                            "IID replacement must be the first native source"
                        )
                    requested.add(wanted.id)
                    actual[observed.id] += 1
                    replacements += int(request["replacement"])
                    stream.write(
                        json.dumps(
                            {
                                "batch": step,
                                "requested": request,
                                "actual": metadata,
                                "label": label,
                                "id_index": identity,
                            }
                        )
                        + "\n"
                    )
                optimizer.zero_grad(set_to_none=True)
                images = batch["pixel_values"].to(runtime.device, non_blocking=True)
                labels = batch["labels"].to(runtime.device, non_blocking=True)
                identities = batch["id_indices"].to(runtime.device, non_blocking=True)
                with runtime.autocast():
                    outputs = self.model.forward_outputs(images)
                    losses = self.model.loss_terms(
                        outputs, labels, id_indices=identities
                    )
                loss = losses["overall"]
                if loss.ndim or not torch.isfinite(loss):
                    atomic_save(
                        output / "nonfinite-batch.pth",
                        {"batch": batch, "outputs": outputs, "losses": losses},
                        torch_format=True,
                    )
                    atomic_save(
                        output / "nonfinite-failure.json",
                        {
                            "batch": step,
                            "labels": labels.tolist(),
                            "identities": identities.tolist(),
                            "reason": "Native EIC requires both classes and norms may be zero; no epsilon/class-balancing repair",
                            "raw": "nonfinite-batch.pth",
                        },
                    )
                    raise ValueError(
                        "IID native objective is nonfinite; complete batch and source receipts preserved"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                ):
                    raise ValueError("Nonfinite IID gradient")
                scaler.step(optimizer)
                scaler.update()
                for key, value in losses.items():
                    components[key] += float(value.detach()) * count
                loss_sum += float(loss.detach()) * count
                images_count += count
        if not images_count:
            raise ValueError("No IID sources trained")
        report = coverage(loader.dataset, set(actual), output)
        request_dir = output / "requested-slots"
        request_dir.mkdir()
        report.update(
            coverage_unit="actual_unique_sources",
            requested_coverage=coverage(loader.dataset, requested, request_dir),
            actual_occurrences=dict(actual),
            model_images=images_count,
            source_replacements=replacements,
        )
        atomic_save(output / "coverage.json", report)
        atomic_save(
            output / "loss-components.json",
            {
                "model_images": images_count,
                "mean": {k: v / images_count for k, v in components.items()},
                "objective": "CE8_sigmoid_margin.2+.05*ArcFace64_margin.4+.1*signedEIC",
            },
        )
        (output / "samples.partial.jsonl").replace(output / "samples.jsonl")
        return loss_sum / images_count, report

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if not isinstance(loader.dataset, IIDDataset) or not isinstance(
            loader.collate_fn, IIDCollator
        ):
            raise ValueError("IID requires its identity dataset and aligned collator")
        output = Path(output_dir)
        atomic_save(
            output / "identity-selection.json", loader.dataset.selection_receipt()
        )
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 11,
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
