"""Command-line manifest preparation and image input checks."""

import argparse
import json
from pathlib import Path

from .config import DataConfig, frames_for
from .dataset import ManifestDataset
from .prepare import prepare_manifest
from .manifest import read_metadata


def prepare_main(argv=None):
    from engine.models import ModelRegistry

    parser = argparse.ArgumentParser(
        description="Index images into a manifest without copying data"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames-per-video", type=int)
    parser.add_argument("--model", choices=[spec.name for spec in ModelRegistry.SPECS],
                        help="Use a detector's manifest preparer")
    parser.add_argument("--source-manifest", type=Path, help="Existing manifest for detector-specific preparation")
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test subset only; never use for reported metrics",
    )
    args = parser.parse_args(argv)
    config = DataConfig(args.config, args.env_file)
    if args.model:
        if args.limit is not None or args.source_manifest is None:
            parser.error("--model requires --source-manifest and cannot use --limit")
        from engine.runtime import task_factory

        model = {"factory": ModelRegistry.get(args.model).factory}
        prepare = getattr(task_factory(model), "prepare_manifest", None)
        if prepare is None:
            parser.error(f"{args.model} does not provide a detector-specific manifest preparer")
        dataset = ManifestDataset(args.source_manifest, config)
        if (dataset.header["dataset"], dataset.header["split"]) != (args.dataset, args.split):
            parser.error("--dataset and --split must match the source manifest")
        report = prepare(dataset, output=args.output, config=config, frames_per_video=args.frames_per_video)
        print(json.dumps(report, indent=2))
        return
    if args.source_manifest is not None:
        parser.error("--source-manifest requires --model")
    media_type = config.media_type(args.dataset)
    frame_count = args.frames_per_video
    if media_type == "image" and frame_count is not None:
        parser.error("--frames-per-video applies to video datasets")
    if frame_count is None and media_type == "video":
        frame_count = frames_for(args.split, config.settings.get("video_sampling", {}))
    suffix = f"-smoke-{args.limit}" if args.limit else ""
    frames = f"-frames-{frame_count}" if frame_count is not None else ""
    if not frames and media_type != "image":
        frames = "-all-frames"
    output = (
        args.output
        or config.manifest_dir() / f"{args.dataset}-{args.split}{frames}{suffix}.jsonl"
    )
    count = prepare_manifest(
        config,
        args.dataset,
        args.split,
        output,
        frames_per_video=frame_count,
        limit=args.limit,
    )
    report = {"manifest": str(output), "samples": count}
    metadata = read_metadata(output)
    if metadata is not None:
        metadata_path = (
            output.parent
            / metadata["manifest_header"]["reconciliation"]["metadata_file"]
        )
        report.update(
            metadata=str(metadata_path),
            available_videos=metadata["available_video_count"],
            stale_metadata_entries=len(metadata["stale_metadata"]),
        )
    print(json.dumps(report, indent=2))


def check_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Decode manifest images and optionally check backbone image processing"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--count", type=int, default=2, help="Images to decode (default: 2)"
    )
    parser.add_argument(
        "--backbone",
        help="Optional model ID for AutoImageProcessor; may download processor config",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")
    dataset = ManifestDataset(args.manifest, DataConfig(args.config, args.env_file))
    samples = [dataset[i] for i in range(min(args.count, len(dataset)))]
    skipped = sum(sample is None for sample in samples)
    samples = [sample for sample in samples if sample is not None]
    report = {
        "checked": len(samples),
        "skipped": skipped,
        "samples": [
            {
                "id": sample["metadata"]["id"],
                "label": sample["label"],
                "size": sample["image"].size,
            }
            for sample in samples
        ],
    }
    if args.backbone:
        from transformers import AutoImageProcessor
        from .loader import ImageCollator

        batch = ImageCollator(AutoImageProcessor.from_pretrained(args.backbone))(
            samples
        )
        report["batch_shape"] = list(batch["pixel_values"].shape) if batch else None
    print(json.dumps(report, indent=2))
