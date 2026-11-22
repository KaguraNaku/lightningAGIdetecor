"""Build reproducible manifests without creating dataset links or copies."""

from itertools import islice
import json
import os
from pathlib import Path
import shutil
import tempfile

from .adapters import image_folders, parquet, video_frames
from .benchmark_splits import read_split
from .config import DataConfig
from .manifest import inventory_digest, sha256, write_json, write_manifest


ADAPTERS = {
    "image_folders": image_folders.prepare,
    "video_frames": video_frames.prepare,
    "parquet": parquet.prepare,
}


def prepare_manifest(
    config: DataConfig,
    name: str,
    split: str,
    output: Path | str,
    *,
    frames_per_video: int | None = None,
    limit: int | None = None,
) -> int:
    """Generate a deterministic manifest; limit is explicitly a smoke-test subset."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    if frames_per_video is not None and (
        type(frames_per_video) is not int or frames_per_video < 1
    ):
        raise ValueError("frames_per_video must be positive or None")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive or None")
    settings = config.dataset(name)
    adapter = settings["adapter"]
    prepare = ADAPTERS.get(adapter)
    if prepare is None:
        raise ValueError(f"Unknown adapter: {adapter}")
    root = settings["root"]
    options = {}
    if "split_format" in settings:
        if adapter != "video_frames":
            raise ValueError("split_format applies to the video_frames adapter")
        options["indexed_split"] = read_split(settings, split)
    records, sources, inventory = prepare(settings, name, split, frames_per_video, **options)
    output = Path(output).expanduser().absolute()
    # A mistaken output path must never overwrite an input shard/split.
    if output.resolve() in {source.resolve() for source in sources}:
        raise ValueError("Manifest output must not overwrite a source file")
    header = {
        "dataset": name,
        "split": split,
        "adapter": adapter,
        "settings": {key: value for key, value in settings.items() if key != "root"},
        "sampling": {
            "frames_per_video": frames_per_video,
            "algorithm": "uniform_endpoints_v1",
            "limit": limit,
        },
        "sources": {
            source.relative_to(root).as_posix(): {
                "size": source.stat().st_size,
                "sha256": sha256(source),
            }
            for source in sources
        },
    }
    if inventory is not None:
        header["reconciliation"] = {
            "policy": inventory["policy"],
            "inventory_sha256": inventory_digest(inventory),
            "reference_video_count": inventory["reference_video_count"],
            "available_video_count": inventory["available_video_count"],
            "stale_metadata_count": len(inventory["stale_metadata"]),
            "files_outside_requested_metadata_count": len(
                inventory["files_outside_requested_metadata"]
            ),
        }
    if limit is not None:
        records = islice(records, limit)
    if inventory is None:
        return write_manifest(output, header, records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        staging = Path(staging)
        initial = staging / "initial.jsonl"
        count = write_manifest(initial, header, records)
        snapshot = sha256(initial)
        metadata_name = f"{output.stem}.{snapshot}.metadata.json"
        header["reconciliation"]["metadata_file"] = metadata_name
        bound = staging / "bound.jsonl"
        with initial.open("rb") as source, bound.open("wb") as target:
            source.readline()
            target.write(
                (
                    json.dumps({"manifest": {"version": 1, **header}}, sort_keys=True)
                    + "\n"
                ).encode()
            )
            shutil.copyfileobj(source, target)
        metadata_path = staging / metadata_name
        write_json(
            metadata_path,
            {
                "version": 1,
                "dataset": name,
                "split": split,
                "manifest_samples": count,
                "manifest_sha256": sha256(bound),
                "manifest_header": {"version": 1, **header},
                "sampling": header["sampling"],
                "original_sources": header["sources"],
                **inventory,
            },
        )
        # The new immutable sidecar is ready before the manifest is published.
        # Existing manifests keep referring to their own snapshot's sidecar.
        os.replace(metadata_path, output.parent / metadata_name)
        os.replace(bound, output)
    return count
