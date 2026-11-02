"""Backbone image processing happens at collation, outside detector modules."""

from dataclasses import dataclass
from functools import partial
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import yaml


@dataclass
class ImageCollator:
    """Use either a batch processor or a PIL-image-to-tensor transform."""

    processor: object = None
    transform: object = None

    def __post_init__(self):
        if (self.processor is None) == (self.transform is None):
            raise ValueError("Supply exactly one of processor or per-image transform")

    def __call__(self, samples):
        samples = [sample for sample in samples if sample is not None]
        if not samples:
            return None
        images = [sample["image"] for sample in samples]
        if self.processor is not None:
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        else:
            pixels = torch.stack([self.transform(image) for image in images])
        return {
            "pixel_values": pixels,
            "labels": torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.long
            ),
            "metadata": [sample["metadata"] for sample in samples],
        }


def seed_worker(worker_id, *, log_level=None):
    if log_level is not None:
        logging.getLogger().setLevel(log_level)
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
    dataset,
    processor=None,
    *,
    collate_fn=None,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    seed=42,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=None,
):
    """Visit each record once, skipping decode failures without replacement."""
    if (processor is None) == (collate_fn is None):
        raise ValueError("Supply exactly one of processor or collate_fn")
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
        worker_init_fn=partial(
            seed_worker, log_level=logging.getLogger().getEffectiveLevel()
        ),
        collate_fn=collate_fn if collate_fn is not None else ImageCollator(processor),
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(
            num_workers > 0
            if persistent_workers is None
            else persistent_workers and num_workers > 0
        ),
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


def create_input_dataset(settings, config):
    """Resolve a manifest input without knowing any model or experiment format."""
    from .dataset import ManifestDataset

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
    return dataset


def create_input_loader(
    settings, *, config, processor=None, collate_fn=None, seed=42, dataset_adapter=None
):
    dataset = create_input_dataset(settings, config)
    if dataset_adapter is not None:
        dataset = dataset_adapter(dataset)
    options = {
        key: settings[key]
        for key in (
            "batch_size",
            "num_workers",
            "shuffle",
            "pin_memory",
            "prefetch_factor",
            "persistent_workers",
        )
        if key in settings
    }
    return create_loader(
        dataset, processor, collate_fn=collate_fn, seed=seed, **options
    )


def create_experiment_loader(
    experiment_path, input_name, *, config=None, processor=None
):
    """Compatibility helper for image-processor experiments; inject other collators directly."""
    from .config import DataConfig

    config = config or DataConfig()
    with Path(experiment_path).open() as stream:
        experiment = yaml.safe_load(stream)
    settings = {**experiment.get("loader", {}), **experiment["inputs"][input_name]}
    if processor is None:
        from transformers import AutoImageProcessor

        model_id = experiment.get("backbone_id") or experiment.get("model", {}).get(
            "backbone_id"
        )
        if not model_id:
            raise ValueError("Pass processor= for experiments without a backbone_id")
        processor = AutoImageProcessor.from_pretrained(model_id)
    return create_input_loader(
        settings,
        config=config,
        processor=processor,
        seed=experiment.get("seed", 42),
    )
