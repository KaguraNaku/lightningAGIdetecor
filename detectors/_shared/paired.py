"""Whole paired-image optimization with separate source/view provenance."""

import json
import logging

import torch

from .trainer import ImageClassificationTrainer
from engine.evaluate import coverage
from engine.runtime import atomic_save

logger = logging.getLogger(__name__)


class PairedImageTrainer(ImageClassificationTrainer):
    fallback_statistic = "fallbacks"

    def validate_view_order(self, sources, metadata):
        """Methods with prescribed view order enforce it before optimization."""

    def training_batches(self, loader, output):
        yield from loader

    def compute_batch_loss(self, batch, images, labels, *, runtime):
        return self.compute_loss(images, labels)

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.model.train()
        expected = {r.id: r for r in loader.dataset.records}
        seen, image_count, fallbacks, loss_sum = set(), 0, 0, 0.0
        receipts = output / "samples.partial.jsonl"
        views = output / "views.partial.jsonl"
        generations = output / "generation.partial.jsonl"
        with (
            receipts.open("w") as stream,
            views.open("w") as view_stream,
            generations.open("w") as generation_stream,
        ):
            for step, batch in enumerate(self.training_batches(loader, output), 1):
                sources = batch["sources"]
                count = len(batch["labels"])
                if (
                    count != 2 * len(sources)
                    or len(batch["pixel_values"]) != count
                    or len(batch["metadata"]) != count
                ):
                    raise ValueError(
                        "Paired requires exactly two aligned model images per source"
                    )
                self.validate_view_order(sources, batch["metadata"])
                batch_sources = {}
                for item in sources:
                    metadata, label, generated = (
                        item["metadata"],
                        item["label"],
                        item["generation"],
                    )
                    record = expected.get(metadata["id"])
                    if (
                        record is None
                        or record.id in seen
                        or record.label != label
                        or record.dataset != metadata["dataset"]
                        or record.split != metadata["split"]
                    ):
                        raise ValueError("Invalid or repeated Paired source identity")
                    generated_label = label if generated["fallback"] else 1
                    if (
                        generated["source_label"] != label
                        or generated["generated_label"] != generated_label
                    ):
                        raise ValueError(
                            "Paired generation/fallback labels differ from source contract"
                        )
                    seen.add(record.id)
                    fallbacks += int(generated["fallback"])
                    batch_sources[record.id] = item
                    stream.write(json.dumps({**metadata, "label": label}) + "\n")
                observed_views = set()
                for index, (metadata, label) in enumerate(
                    zip(batch["metadata"], batch["labels"].tolist(), strict=True)
                ):
                    item = batch_sources.get(metadata["id"])
                    kind = metadata.get("view")
                    identity = metadata["id"], kind
                    if (
                        item is None
                        or kind not in ("generated", "original")
                        or identity in observed_views
                    ):
                        raise ValueError(
                            "Invalid, unrelated or repeated Paired view identity"
                        )
                    if any(
                        metadata[k] != item["metadata"][k]
                        for k in ("dataset", "split", "video_id")
                    ):
                        raise ValueError("Paired view metadata differs from its parent")
                    desired = (
                        item["label"]
                        if kind == "original"
                        else item["generation"]["generated_label"]
                    )
                    if label != desired:
                        raise ValueError(
                            "Paired shuffled view label differs from its parent/generation"
                        )
                    observed_views.add(identity)
                    view_stream.write(
                        json.dumps(
                            {
                                **metadata,
                                "batch": step,
                                "model_index": index,
                                "label": label,
                            }
                        )
                        + "\n"
                    )
                generation_stream.write(
                    json.dumps(
                        {
                            "batch": step,
                            "rng": batch["generation_rng"],
                            "sources": sources,
                        }
                    )
                    + "\n"
                )
                optimizer.zero_grad(set_to_none=True)
                images = batch["pixel_values"].to(runtime.device, non_blocking=True)
                labels = batch["labels"].to(runtime.device, non_blocking=True)
                with runtime.autocast():
                    loss = self.compute_batch_loss(
                        batch, images, labels, runtime=runtime
                    )
                if loss.ndim or not torch.isfinite(loss):
                    raise ValueError("Nonfinite or nonscalar Paired loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                ):
                    raise ValueError("Nonfinite Paired gradient")
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * count
                image_count += count
                if step % 100 == 0:
                    logger.info(
                        "Paired batch %d/%d, source images%d, model images%d, loss%.6f",
                        step,
                        len(loader),
                        len(seen),
                        image_count,
                        loss_sum / image_count,
                    )
        if not seen:
            raise ValueError(
                "No Paired source images successfully decoded for training"
            )
        report = {
            **coverage(loader.dataset, seen, output),
            "model_images": image_count,
            "generated_views": len(seen),
            "original_views": len(seen),
            self.fallback_statistic: fallbacks,
        }
        atomic_save(output / "coverage.json", report)
        receipts.replace(output / "samples.jsonl")
        views.replace(output / "views.jsonl")
        generations.replace(output / "generation.jsonl")
        return loss_sum / image_count, report
