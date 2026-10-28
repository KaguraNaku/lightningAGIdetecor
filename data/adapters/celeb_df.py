"""Celeb-DF official test list; its labels use real=1, fake=0."""

from pathlib import PurePosixPath

from data.manifest import relative_path
from data.reconcile import Video, frame_directories, materialize, reconcile


def inventory(settings, name, split):
    if split != "test":
        raise ValueError("Celeb-DF adapter requires the official test split")
    root = settings["root"]
    entries = []
    for line in (root / "List_of_testing_videos.txt").read_text().splitlines():
        if not line.strip():
            continue
        label, video = line.split()
        path = PurePosixPath(relative_path(video))
        if len(path.parts) != 2 or label not in ("0", "1"):
            raise ValueError(f"Invalid Celeb-DF test entry: {line}")
        category = path.parts[0]
        expected = "0" if category == "Celeb-synthesis" else "1"
        if (
            category not in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
            or label != expected
        ):
            raise ValueError(f"Inconsistent Celeb-DF label: {line}")
        entries.append((video, 1 - int(label), category, path.stem))
    expected = [
        Video(
            root / category / "frames" / video,
            f"{category}/{video}",
            label,
            category,
            source_key=source,
        )
        for source, label, category, video in sorted(entries)
    ]
    available = [
        path
        for category in ("Celeb-real", "Celeb-synthesis", "YouTube-real")
        for path in frame_directories(root / category / "frames")
    ]
    return reconcile(root, expected, available)


def build(settings, name, split, frame_count):
    videos, _ = inventory(settings, name, split)
    yield from materialize(videos, settings["root"], name, split, frame_count)
