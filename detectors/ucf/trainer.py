"""Whole ordered real/fake pairs with separate fake-slot and real-occurrence coverage."""

from collections import Counter
import json
from pathlib import Path

import torch

from detectors._shared.trainer import ImageClassificationTrainer
from engine.evaluate import coverage
from engine.runtime import atomic_save
from .dataset import UCFDataset, UCFCollator, UCFDataError, metadata


class UCFTrainer(ImageClassificationTrainer):
    @staticmethod
    def validate_inputs(loader, validation_loader, monitor, *, training_labels=None):
        ImageClassificationTrainer.validate_inputs(
            loader, validation_loader, monitor, training_labels={1}
        )
        # Validation must also exclude the sampled real pool, not just fake slots.
        train, val = loader.dataset, validation_loader.dataset
        identity = lambda row: (row.storage, row.path, row.row_group, row.row_index)
        if {identity(row) for row in train.all_records} & {
            identity(row) for row in val.records
        }:
            raise ValueError("UCF entire source pool overlaps validation images")
        if {row.video_id for row in train.all_records if row.video_id} & {
            row.video_id for row in val.records if row.video_id
        }:
            raise ValueError("UCF entire source pool overlaps validation videos")

    def training_batches(self, loader, output):
        try:
            yield from loader
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            if receipt is None and UCFDataError.marker in str(error):
                try:
                    receipt = json.JSONDecoder().raw_decode(
                        str(error).split(UCFDataError.marker, 1)[1]
                    )[0]
                except (ValueError, TypeError):
                    pass
            atomic_save(
                output / "source-failure.json",
                receipt or {"error": f"{type(error).__name__}: {error}"},
            )
            raise

    def train_epoch(self, loader, optimizer, scaler, *, runtime, output):
        self.model.train()
        expected = {row.id: row for row in loader.dataset.records}
        reals = {row.id: row for row in loader.dataset.real_records}
        seen, occurrences, components = set(), Counter(), Counter()
        loss_sum, image_count = 0.0, 0
        with (output / "samples.partial.jsonl").open("w") as stream:
            for step, batch in enumerate(self.training_batches(loader, output), 1):
                pairs, labels, specific = (
                    batch["pairs"],
                    batch["labels"],
                    batch["specific_labels"],
                )
                count, size = len(labels), len(pairs)
                if (
                    not size
                    or count != 2 * size
                    or len(batch["pixel_values"]) != count
                    or len(batch["metadata"]) != count
                    or specific.shape != (count,)
                ):
                    raise ValueError(
                        "UCF requires whole aligned real/fake/specific-label pairs"
                    )
                if labels.tolist() != [0] * size + [1] * size:
                    raise ValueError("UCF model order must be all real then all fake")
                for index, pair in enumerate(pairs):
                    fake, real = (
                        expected.get(pair["fake_id"]),
                        reals.get(pair["real_id"]),
                    )
                    if fake is None or real is None or fake.id in seen:
                        raise ValueError(
                            "UCF invalid/repeated indexed fake or unknown sampled real"
                        )
                    if batch["metadata"][index] != metadata(real) or batch["metadata"][
                        size + index
                    ] != metadata(fake):
                        raise ValueError(
                            "UCF paired provenance differs from source inventory"
                        )
                    if (
                        specific[index].item() != 0
                        or specific[size + index].item()
                        != fake.metadata["specific_label"]
                    ):
                        raise ValueError(
                            "UCF specific labels differ from paired sources"
                        )
                    seen.add(fake.id)
                    occurrences[real.id] += 1
                    stream.write(
                        json.dumps(
                            {
                                "batch": step,
                                "pair": pair,
                                "real": metadata(real),
                                "fake": metadata(fake),
                            }
                        )
                        + "\n"
                    )
                with runtime.time_region("fit/batch", samples=count):
                    optimizer.zero_grad(set_to_none=True)
                    images = batch["pixel_values"].to(runtime.device, non_blocking=True)
                    labels = labels.to(runtime.device, non_blocking=True)
                    specific = specific.to(runtime.device, non_blocking=True)
                    with runtime.autocast():
                        outputs = self.model.forward_outputs(images)
                        losses = self.model.loss_terms(
                            outputs, images, labels, specific
                        )
                    loss = losses["overall"]
                    if loss.ndim or not torch.isfinite(loss):
                        atomic_save(
                            output / "nonfinite-batch.pth",
                            {"batch": batch, "outputs": outputs, "losses": losses},
                            torch_format=True,
                        )
                        raise ValueError(
                            "Nonfinite UCF native objective; complete paired batch preserved"
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in self.model.parameters()
                    ):
                        raise ValueError("Nonfinite UCF gradient")
                    scaler.step(optimizer)
                    scaler.update()
                for key, value in losses.items():
                    components[key] += float(value.detach()) * count
                image_count += count
                loss_sum += float(loss.detach()) * count
        if seen != set(expected) or not image_count:
            raise ValueError(
                "UCF indexed fake coverage is incomplete; no dropped or repeated slots allowed"
            )
        (output / "samples.partial.jsonl").replace(output / "samples.jsonl")
        report = coverage(loader.dataset, seen, output)
        report.update(
            model_images=image_count,
            indexed_fake_count=len(seen),
            sampled_real_occurrences=sum(occurrences.values()),
            sampled_real_unique=len(occurrences),
            sampled_real_counts=dict(occurrences),
            sampled_real_pool=len(reals),
            effective_batch="2*source_pair_batch",
            collation="all_real_then_all_fake",
        )
        atomic_save(output / "coverage.json", report)
        atomic_save(
            output / "loss-components.json",
            {
                "model_images": image_count,
                "mean": {key: value / image_count for key, value in components.items()},
                "objective": "commonCE+.1*specificCE+.3*fourL1+.05*contrastive_margin3",
            },
        )
        return loss_sum / image_count, report

    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        if not isinstance(loader.dataset, UCFDataset) or not isinstance(
            loader.collate_fn, UCFCollator
        ):
            raise ValueError(
                "UCF training requires native ordered pairing and specific labels"
            )
        output = Path(output_dir)
        atomic_save(output / "pair-transport.json", loader.dataset.selection_receipt())
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={
                "epochs": 6,
                "learning_rate": 2e-4,
                "weight_decay": 5e-4,
                **settings,
            },
            output_dir=output,
        )
