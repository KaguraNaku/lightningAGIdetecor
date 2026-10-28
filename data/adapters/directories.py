"""Explicit directory protocols for frame datasets and independent images."""

import json
from pathlib import PurePosixPath

from data.manifest import Sample, relative_path
from data.reconcile import Video, frame_directories, materialize, reconcile
from .frames import EXTENSIONS, natural_key


def inventory(settings, name, split):
    protocol = settings.get("splits", {}).get(split)
    if not protocol or "groups" not in protocol or "protocol" not in protocol:
        raise ValueError(f"{name}: explicit groups and protocol required for {split}")
    root = settings["root"]
    expected, available = [], []
    for group in protocol["groups"]:
        pattern = relative_path(group["pattern"])
        if "**" in pattern:
            raise ValueError("Use bounded directory patterns, not recursive **")
        directories = sorted(
            (
                p
                for p in root.glob(pattern)
                if p.is_dir() and not p.name.startswith(".")
            ),
            key=lambda p: p.as_posix(),
        )
        for directory in directories:
            video_id = directory.relative_to(root).as_posix()
            expected.append(
                Video(directory, video_id, group["label"], group["category"])
            )
            available.append(directory)
    return reconcile(root, expected, available)


def build(settings, name, split, frame_count):
    videos, _ = inventory(settings, name, split)
    yield from materialize(videos, settings["root"], name, split, frame_count)


def dfdc_inventory(settings, name, split):
    protocol = settings.get("splits", {}).get(split)
    if not protocol:
        raise ValueError(f"{name}: unavailable split {split}")
    root = settings["root"]
    entries = json.loads((root / relative_path(protocol["metadata"])).read_text())
    expected = []
    for video, attributes in sorted(entries.items()):
        video_path = PurePosixPath(relative_path(video))
        if len(video_path.parts) != 1:
            raise ValueError(f"Expected video basename in DFDC metadata: {video}")
        label = attributes.get("is_fake")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Invalid DFDC is_fake for {video}: {label}")
        directory = root / relative_path(protocol["frames"]) / video_path.stem
        expected.append(
            Video(
                directory,
                video_path.stem,
                label,
                "fake" if label else "real",
                source_key=video,
                source_metadata=attributes,
            )
        )
    return reconcile(
        root, expected, frame_directories(root / relative_path(protocol["frames"]))
    )


def dfdc(settings, name, split, frame_count):
    videos, _ = dfdc_inventory(settings, name, split)
    yield from materialize(videos, settings["root"], name, split, frame_count)


def universal(settings, name, split, frame_count):
    if frame_count is not None:
        raise ValueError(
            "UniversalFakeDetect contains independent images; do not use frames_per_video"
        )
    partition = settings.get("splits", {}).get(split)
    if not partition:
        raise ValueError(f"{name}: unavailable split {split}")
    root = settings["root"]
    directory = root / relative_path(partition)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for category in sorted(directory.iterdir(), key=natural_key):
        if (
            category.name.startswith(".")
            or category.name in settings.get("ignore_dirs", [])
            or not category.is_dir()
        ):
            continue

        def has_classes(path):
            return any((path / folder).is_dir() for folder in ("0_real", "1_fake"))

        # Some generators contain domains (e.g. cyclegan/apple/0_real).
        # Shared real pools and fake-only generators legitimately have one class.
        leaves = (
            [category]
            if has_classes(category)
            else [
                p
                for p in sorted(category.iterdir(), key=natural_key)
                if p.is_dir() and not p.name.startswith(".") and has_classes(p)
            ]
        )
        if not leaves:
            raise ValueError(
                f"No image class folders found under {category}; declare non-data directories in ignore_dirs"
            )
        for leaf in leaves:
            for label, folder in ((0, "0_real"), (1, "1_fake")):
                images = leaf / folder
                if not images.is_dir():
                    continue
                files = sorted(
                    (
                        p
                        for p in images.iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS
                    ),
                    key=natural_key,
                )
                if not files:
                    continue
                for image in files:
                    path = image.relative_to(root).as_posix()
                    yield Sample(
                        id=f"{name}:{path}",
                        dataset=name,
                        label=label,
                        split=split,
                        storage="image",
                        path=path,
                        category=category.name,
                        metadata={"subcategory": leaf.name} if leaf != category else {},
                    )
