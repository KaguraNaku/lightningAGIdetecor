"""Build effective video metadata from existing frame directories."""

from dataclasses import dataclass, field
from pathlib import Path

from .adapters.image_folders import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class Video:
    directory: Path
    video_id: str
    label: int
    category: str
    source_key: str = ""
    source_metadata: dict = field(default_factory=dict)


def reconcile(
    root: Path, expected: list[Video], available: list[Path]
) -> tuple[list[Video], dict]:
    reference = {}
    for video in expected:
        if video.directory in reference:
            raise ValueError(f"Duplicate metadata directory: {video.directory}")
        reference[video.directory] = video
    actual = set(available)
    active = []
    stale = []
    unmatched = []
    for directory in available:
        if directory not in reference:
            unmatched.append(directory.relative_to(root).as_posix())
            continue
        video = reference[directory]
        if any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in directory.iterdir()
        ):
            active.append(video)
        else:
            stale.append(
                {
                    "video_id": video.video_id,
                    "source_key": video.source_key,
                    "reason": "no_supported_image_files",
                }
            )
    for video in expected:
        if video.directory not in actual:
            stale.append(
                {
                    "video_id": video.video_id,
                    "source_key": video.source_key,
                    "reason": "frames_directory_missing",
                }
            )
    metadata = {
        video.video_id: {
            "label": video.label,
            "category": video.category,
            "frames": video.directory.relative_to(root).as_posix(),
            "source_key": video.source_key,
            "source_metadata": video.source_metadata,
        }
        for video in active
    }
    return active, {
        "policy": "files_first",
        "reference_video_count": len(expected),
        "available_video_count": len(active),
        "stale_metadata": stale,
        "files_outside_requested_metadata": unmatched,
        "available_video_metadata": metadata,
    }
