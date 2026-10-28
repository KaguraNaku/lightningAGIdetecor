"""Shared deterministic frame enumeration; never recursively follows links."""

from pathlib import Path
import re

from data.manifest import Sample

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def natural_key(path: Path) -> tuple:
    parts = tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", path.name)
    )
    return parts, path.name


def video_frames(
    root: Path,
    directory: Path,
    *,
    dataset: str,
    split: str,
    label: int,
    video_id: str,
    category: str,
    frame_count: int | None,
):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing frames for video {video_id}: {directory}")
    frames = sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ),
        key=natural_key,
    )
    if not frames:
        raise ValueError(f"No frames for video {video_id}: {directory}")
    if frame_count is not None and len(frames) > frame_count:
        # Even spacing including endpoints; one frame selects the first.
        indices = [
            i * (len(frames) - 1) // max(frame_count - 1, 1) for i in range(frame_count)
        ]
        frames = [frames[i] for i in indices]
    for image in frames:
        path = image.relative_to(root).as_posix()
        yield Sample(
            id=f"{dataset}:{path}",
            dataset=dataset,
            label=label,
            split=split,
            storage="image",
            path=path,
            video_id=video_id,
            frame_index=int(image.stem) if image.stem.isdigit() else None,
            category=category,
        )
