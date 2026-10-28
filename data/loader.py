"""Backbone image processing happens at collation, outside detector modules."""

from dataclasses import dataclass
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import yaml


@dataclass
class ImageCollator:
    processor: object

    def __call__(self, samples):
        samples = [sample for sample in samples if sample is not None]
        if not samples:
            return None
        inputs = self.processor(
            images=[sample["image"] for sample in samples], return_tensors="pt"
        )
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.long
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class RowGroupSampler(Sampler):
    """Shuffle groups and their rows, keeping each group contiguous for image IO."""

    def __init__(self, records, generator):
        groups = {}
        for index, record in enumerate(records):
            key = (
                (record.path, record.row_group)
                if record.storage == "parquet"
                else ("image", index)
            )
            groups.setdefault(key, []).append(index)
        self.groups = list(groups.values())
        self.generator = generator
        self.size = len(records)

    def __len__(self):
        return self.size

    def __iter__(self):
        for group_index in torch.randperm(
            len(self.groups), generator=self.generator
        ).tolist():
            group = self.groups[group_index]
            for row in torch.randperm(len(group), generator=self.generator).tolist():
                yield group[row]


class ImageDataLoader(DataLoader):
    """Omit empty batches after failed image records have been filtered out."""

    def __iter__(self):
        for batch in super().__iter__():
            if batch is not None:
                yield batch


def create_loader(
    dataset, processor, *, batch_size=32, shuffle=False, num_workers=0, seed=42
):
    """Visit each record once, skipping decode failures without replacement."""
    generator = torch.Generator().manual_seed(seed)
    records = getattr(dataset, "records", [])
    sampler = None
    if shuffle and any(record.storage == "parquet" for record in records):
        sampler = RowGroupSampler(records, generator)
        shuffle = False
    return ImageDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        generator=generator,
        worker_init_fn=seed_worker,
        collate_fn=ImageCollator(processor),
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def create_experiment_loader(
    experiment_path, input_name, *, config=None, processor=None
):
    """Resolve one named image input from an experiment, without running a model."""
    from .config import DataConfig
    from .dataset import ManifestDataset

    config = config or DataConfig()
    with Path(experiment_path).open() as stream:
        experiment = yaml.safe_load(stream)
    settings = experiment["inputs"][input_name]
    value = settings["manifest"].replace("${MANIFEST_ROOT}", str(config.manifest_dir()))
    path = Path(config.expand(value)).expanduser()
    if not path.is_absolute():
        raise ValueError("Experiment manifest path must be absolute")
    dataset = ManifestDataset(
        path,
        config,
        decode_error_patience=settings.get("decode_error_patience", 10),
    )
    if (
        dataset.header["dataset"] != settings["dataset"]
        or dataset.header["split"] != settings["split"]
    ):
        raise ValueError("Experiment dataset/split does not match its manifest")
    if dataset.header["sampling"].get("limit") is not None:
        raise ValueError("Experiment inputs must not use limited smoke-test manifests")
    if processor is None:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(experiment["backbone_id"])
    return create_loader(
        dataset,
        processor,
        batch_size=settings.get("batch_size", 32),
        num_workers=settings.get("num_workers", 0),
        shuffle=settings.get("shuffle", False),
        seed=experiment.get("seed", 42),
    )
