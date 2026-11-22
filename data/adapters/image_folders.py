"""Read image files from class folders declared in the dataset config."""

from pathlib import Path
import re

from data.manifest import Sample, relative_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def natural_key(path: Path) -> tuple:
    parts = tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", path.name)
    )
    return parts, path.name


def image_files(directory):
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )


def prepare(settings, name, split, frame_count):
    if frame_count is not None:
        raise ValueError("Omit frames_per_video for independent images in class folders")
    partition = settings.get("splits", {}).get(split)
    if partition is None:
        raise ValueError(f"{name}: unavailable split {split}")
    root = settings["root"]
    directory = root if partition == "." else root / relative_path(partition)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    classes = sorted(
        (label, relative_path(folder))
        for folder, label in settings["class_folders"].items()
    )
    depth = settings.get("group_depth", 0)
    if type(depth) is not int or depth < 0:
        raise ValueError("group_depth must be a nonnegative integer")
    groups = _class_groups(directory, classes, depth, set(settings.get("ignore_dirs", [])))
    return _records(groups, root, directory, name, split, classes), (), None


def _class_groups(directory, classes, depth, ignored):
    if any((directory / folder).is_dir() for _, folder in classes):
        yield directory
    elif depth:
        for child in sorted(directory.iterdir(), key=natural_key):
            if child.is_dir() and not child.name.startswith(".") and child.name not in ignored:
                yield from _class_groups(child, classes, depth - 1, ignored)


def _records(groups, root, partition, name, split, classes):
    for group in groups:
        parts = group.relative_to(partition).parts
        category = parts[0] if parts else ""
        metadata = {"subcategory": "/".join(parts[1:])} if len(parts) > 1 else {}
        for label, folder in classes:
            directory = group / folder
            if not directory.is_dir():
                continue
            for image in image_files(directory):
                path = image.relative_to(root).as_posix()
                yield Sample(
                    id=f"{name}:{path}", dataset=name, label=label, split=split,
                    storage="image", path=path, category=category, metadata=metadata,
                )
