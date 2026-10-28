"""Official FF++ identity splits, retaining both source identities of fakes."""

import json

from data.reconcile import Video, frame_directories, materialize, reconcile


def inventory(settings, name, split):
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
    return reconcile(root, expected, available)


def build(settings, name, split, frame_count):
    videos, _ = inventory(settings, name, split)
    yield from materialize(videos, settings["root"], name, split, frame_count)
