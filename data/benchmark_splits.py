"""Read official benchmark split metadata into video identities."""

import json
from pathlib import PurePosixPath

from .manifest import relative_path
from .reconcile import Video, frame_directories


def _ffpp(settings, split):
    root = settings["root"]
    split_file = root / f"{split}.json"
    pairs = json.loads(split_file.read_text())
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(
            not isinstance(p, list)
            or len(p) != 2
            or any(not isinstance(v, str) or not v.isdigit() for v in p)
            for p in pairs
        )
    ):
        raise ValueError(f"Invalid official FF++ split: {split_file}")
    excluded = settings.get("exclude_ids", [])
    if not isinstance(excluded, list) or any(
        not isinstance(v, str) or not v.isdigit() for v in excluded
    ):
        raise ValueError("FF++ exclude_ids must be a list of string video identities")
    excluded = set(excluded)
    identities = {v for pair in pairs for v in pair} - excluded
    directed_pairs = sorted(
        {
            f"{a}_{b}"
            for pair in pairs
            if not excluded.intersection(pair)
            for a, b in (pair, pair[::-1])
        }
    )
    compression = settings.get("compression", "c23")
    categories = [
        ("youtube", 0, root / f"original_sequences/youtube/{compression}/frames")
    ]
    categories += [
        (method, 1, root / f"manipulated_sequences/{method}/{compression}/frames")
        for method in settings["methods"]
    ]
    expected, available = [], []
    for category, label, directory in categories:
        ids = sorted(identities) if label == 0 else directed_pairs
        expected.extend(
            Video(
                directory / video,
                f"{category}/{video}",
                label,
                category,
                source_key=video,
            )
            for video in ids
        )
        available.extend(frame_directories(directory))
    return expected, available, (split_file,)


def _celeb_df(settings, split):
    if split != "test":
        raise ValueError("Celeb-DF requires the official test split")
    root = settings["root"]
    test_list = root / "List_of_testing_videos.txt"
    entries = []
    for line in test_list.read_text().splitlines():
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
    return expected, available, (test_list,)


def _dfdc(settings, split):
    protocol = settings.get("splits", {}).get(split)
    if not protocol:
        raise ValueError(f"DFDC has no metadata for split {split}")
    root = settings["root"]
    metadata_file = root / relative_path(protocol["metadata"])
    entries = json.loads(metadata_file.read_text())
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
    available = frame_directories(root / relative_path(protocol["frames"]))
    return expected, available, (metadata_file,)


_SPLITS = {"ffpp": _ffpp, "celeb_df": _celeb_df, "dfdc": _dfdc}


def read_split(settings, split):
    format_name = settings["split_format"]
    reader = _SPLITS.get(format_name)
    if reader is None:
        raise ValueError(f"Unknown video split format: {format_name}")
    return reader(settings, split)
