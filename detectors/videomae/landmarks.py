"""Bind native filesystem landmark presence/bytes before clip decoding."""

from pathlib import Path

import numpy as np

from data.manifest import relative_path, sha256


def native_path(root, frame_path):
    # Apply native literal replacements to the actual absolute image string.
    # Reject escaping roots rather than silently choosing a different sidecar.
    source = str(Path(root) / relative_path(frame_path))
    target = Path(source.replace("frames", "landmarks").replace(".png", ".npy"))
    try:
        return relative_path(target.relative_to(root).as_posix())
    except ValueError as error:
        raise ValueError("Native VideoMAE landmark mapping must remain inside the dataset root") from error


def bind(root, record):
    name = native_path(root, record.path)
    path = Path(root) / name
    present = path.exists()
    descriptor = {"path": name, "sha256": sha256(path) if present else None,
                  "bytes": path.stat().st_size if present else None}
    expected = record.metadata.get("videomae_landmarks")
    if expected is not None and (
        not isinstance(expected, dict)
        or set(expected) not in ({"path", "sha256"}, {"path", "sha256", "bytes"})
        or any(expected[key] != descriptor[key] for key in expected)
    ):
        raise ValueError(f"VideoMAE landmark snapshot differs from declared metadata: {record.id}")
    return descriptor


def load(root, descriptor):
    path = Path(root) / descriptor["path"]
    if descriptor["sha256"] is None:
        if path.exists():
            raise ValueError("VideoMAE previously missing landmark file appeared after binding")
        points = np.zeros((81, 2))
    else:
        if (not path.exists() or path.stat().st_size != descriptor["bytes"]
                or sha256(path) != descriptor["sha256"]):
            raise ValueError("VideoMAE landmark bytes changed after binding")
        # Native filesystem mode does not rescale or cast NPY coordinates.
        # np.load's default pickle policy is False; malformed input stays fatal.
        points = np.load(path, allow_pickle=False)
    return points, {**descriptor, "shape": list(points.shape), "dtype": str(points.dtype),
                    "native_missing_zero_fill": descriptor["sha256"] is None}
