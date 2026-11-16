"""Command-line manifest preparation and image input checks."""

import argparse
import json
from pathlib import Path

from .config import DataConfig
from .dataset import ManifestDataset
from .prepare import prepare_manifest
from .manifest import read_metadata


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Index images into a manifest without copying data"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames-per-video", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test subset only; never use for reported metrics",
    )
    args = parser.parse_args(argv)
    config = DataConfig(args.config, args.env_file)
    suffix = f"-smoke-{args.limit}" if args.limit else ""
    frames = (
        f"-frames-{args.frames_per_video}" if args.frames_per_video else "-all-frames"
    )
    output = (
        args.output
        or config.manifest_dir() / f"{args.dataset}-{args.split}{frames}{suffix}.jsonl"
    )
    count = prepare_manifest(
        config,
        args.dataset,
        args.split,
        output,
        frames_per_video=args.frames_per_video,
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument(
        "--experiment", type=Path, help="Load one image batch using an experiment YAML"
    )
    parser.add_argument(
        "--input", help="Named experiment input; defaults to calibration"
    )
    parser.add_argument(
        "--count", type=int, help="Images to decode in manifest mode (default: 2)"
    )
    parser.add_argument(
        "--backbone",
        help="Optional model ID for AutoImageProcessor; may download processor config",
    )
    args = parser.parse_args(argv)
    if args.count is not None and args.count < 1:
        parser.error("--count must be positive")
    if args.experiment:
        if args.count is not None or args.backbone:
            parser.error(
                "--count and --backbone apply to --manifest; experiment mode uses its YAML settings"
            )
        from .loader import create_experiment_loader

        input_name = args.input or "calibration"
        loader = create_experiment_loader(
            args.experiment, input_name, config=DataConfig(args.config, args.env_file)
        )
        batch = next(iter(loader))
        print(
            json.dumps(
                {
                    "input": input_name,
                    "checked": len(batch["labels"]),
                    "batch_shape": list(batch["pixel_values"].shape),
                    "labels": batch["labels"].tolist(),
                    "first_sample": batch["metadata"][0],
                },
                indent=2,
            )
        )
        return
    if args.input:
        parser.error("--input requires --experiment")
    dataset = ManifestDataset(args.manifest, DataConfig(args.config, args.env_file))
    samples = [dataset[i] for i in range(min(args.count or 2, len(dataset)))]
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
