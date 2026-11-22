"""Index video frame directories and sample frames at evenly spaced positions."""

from data.manifest import Sample, relative_path
from data.reconcile import Video, reconcile
from .image_folders import image_files


def prepare(settings, name, split, frame_count):
    root = settings["root"]
    expected = _directory_groups(settings, name, split)
    available = [video.directory for video in expected]
    videos, inventory = reconcile(root, expected, available)
    return _records(videos, root, name, split, frame_count), (), inventory


def _directory_groups(settings, name, split):
    groups = settings.get("splits", {}).get(split, {}).get("groups")
    if not groups:
        raise ValueError(f"{name}: video frame directories require groups for {split}")
    root = settings["root"]
    videos = []
    for group in groups:
        pattern = relative_path(group["pattern"])
        if "**" in pattern:
            raise ValueError("Use bounded directory patterns for video frames")
        for directory in sorted(root.glob(pattern), key=lambda path: path.as_posix()):
            if directory.is_dir() and not directory.name.startswith("."):
                video_id = directory.relative_to(root).as_posix()
                videos.append(Video(directory, video_id, group["label"], group.get("category", "")))
    return videos


def _records(videos, root, name, split, frame_count):
    for video in videos:
        frames = image_files(video.directory)
        if not frames:
            raise ValueError(f"No frames for video {video.video_id}: {video.directory}")
        if frame_count is not None and len(frames) > frame_count:
            indices = [
                i * (len(frames) - 1) // max(frame_count - 1, 1)
                for i in range(frame_count)
            ]
            frames = [frames[i] for i in indices]
        for image in frames:
            path = image.relative_to(root).as_posix()
            yield Sample(
                id=f"{name}:{path}", dataset=name, label=video.label, split=split,
                storage="image", path=path, video_id=video.video_id, category=video.category,
                frame_index=int(image.stem) if image.stem.isdigit() else None,
            )
