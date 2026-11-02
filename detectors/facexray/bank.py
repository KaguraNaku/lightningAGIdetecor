"""Pinned, mapped landmark/neighbor banks bound to a complete train manifest."""

import json
import os
from pathlib import Path
import random

import numpy as np

from data.manifest import relative_path, sha256


ARRAYS = {
    "landmarks": ("landmarks.npy", np.dtype("float32")),
    "neighbor_offsets": ("neighbor_offsets.npy", np.dtype("int64")),
    "neighbor_indices": ("neighbor_indices.npy", np.dtype("int32")),
    "neighbor_present": ("neighbor_present.npy", np.dtype("bool")),
}


def bank_identity(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_bank(output, *, manifest_path, records, landmarks, neighbors, origin):
    """Compile explicit per-member inputs; missing rows mean native self fallback.

    neighbors maps member row to an ordered, nonempty row-index list. It is never
    truncated, deduplicated or silently restricted to a subset of a manifest.
    """
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    count = len(records)
    if not count or len(landmarks) != count:
        raise ValueError("FaceXray bank requires one landmark array per manifest row")
    points = np.stack(landmarks)
    if (
        points.shape != (count, 81, 2)
        or points.dtype != np.float32
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "FaceXray bank preserves finite native float32(81,2) landmarks"
        )
    if not all(type(index) is int and 0 <= index < count for index in neighbors):
        raise ValueError("FaceXray neighbor parent is outside its training manifest")
    offsets, indices, present = [0], [], []
    for index in range(count):
        exists = index in neighbors
        values = list(neighbors[index]) if exists else []
        if exists and not values:
            raise ValueError(
                "FaceXray empty neighbor entry is invalid; it is not self fallback"
            )
        if any(type(value) is not int or not 0 <= value < count for value in values):
            raise ValueError(
                "FaceXray foreground is outside the complete training manifest"
            )
        present.append(exists)
        indices.extend(values)
        offsets.append(len(indices))
    arrays = {
        "landmarks": points,
        "neighbor_offsets": np.asarray(offsets, dtype=np.int64),
        "neighbor_indices": np.asarray(indices, dtype=np.int32),
        "neighbor_present": np.asarray(present, dtype=np.bool_),
    }
    identities = {}
    for key, (name, _) in ARRAYS.items():
        path = output / name
        np.save(path, arrays[key], allow_pickle=False)
        identities[key] = {
            "path": name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    header = {
        "version": 1,
        "recipe": "ffblend81_explicit_manifest_members_v1",
        "manifest_sha256": sha256(Path(manifest_path)),
        "member_ids": [record.id for record in records],
        "arrays": identities,
        "origin": origin,
        "selection_policy": "All explicit manifest backgrounds, mixed0/1 labels retained; complete ordered neighbor lists must resolve inside the SAME train inventory. No hidden filter or coordinate clamp.",
    }
    path = output / "bank.json"
    path.write_text(json.dumps(header, indent=2, allow_nan=False) + "\n")
    return bank_identity(path)


class FaceBank:
    def __init__(self, identity, dataset):
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError("FaceXray requires a complete compiled bank identity")
        self.identity = dict(identity)
        self._pid, self._arrays, self._stats = None, {}, {}
        path = Path(identity["path"])
        if (
            path.stat().st_size != identity["bytes"]
            or sha256(path) != identity["sha256"]
        ):
            raise ValueError("FaceXray compiled bank header changed")
        self.header = json.loads(path.read_text())
        if (
            self.header.get("version") != 1
            or self.header.get("recipe") != "ffblend81_explicit_manifest_members_v1"
            or set(self.header.get("arrays", {})) != set(ARRAYS)
            or dataset.header["split"] != "train"
            or self.header.get("manifest_sha256") != sha256(dataset.manifest_path)
            or self.header.get("member_ids")
            != [record.id for record in dataset.records]
        ):
            raise ValueError(
                "FaceXray bank must bind every row of this exact train manifest"
            )
        self._open()

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_arrays": {}, "_stats": {}}

    def _open(self):
        if self._pid == os.getpid():
            for path, expected in self._stats.items():
                info = Path(path).stat()
                if (info.st_size, info.st_mtime_ns) != expected:
                    raise ValueError("FaceXray mapped bank changed during training")
            return
        root = Path(self.identity["path"]).parent
        header_path = Path(self.identity["path"])
        info = header_path.stat()
        if (
            info.st_size != self.identity["bytes"]
            or sha256(header_path) != self.identity["sha256"]
        ):
            raise ValueError(
                "FaceXray compiled bank header changed before worker loading"
            )
        arrays, stats = {}, {str(header_path): (info.st_size, info.st_mtime_ns)}
        for key, (_, dtype) in ARRAYS.items():
            descriptor = self.header["arrays"][key]
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "path",
                "sha256",
                "bytes",
            }:
                raise ValueError("FaceXray array requires path/hash/size")
            path = root / relative_path(descriptor["path"])
            info = path.stat()
            if (
                info.st_size != descriptor["bytes"]
                or sha256(path) != descriptor["sha256"]
            ):
                raise ValueError(f"FaceXray bank array changed: {key}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != dtype:
                raise ValueError(f"FaceXray bank array dtype changed: {key}")
            arrays[key] = array
            stats[str(path)] = (info.st_size, info.st_mtime_ns)
        count = len(self.header["member_ids"])
        offsets, indices, present = (
            arrays[k]
            for k in ["neighbor_offsets", "neighbor_indices", "neighbor_present"]
        )
        if (
            arrays["landmarks"].shape != (count, 81, 2)
            or offsets.shape != (count + 1,)
            or indices.ndim != 1
            or present.shape != (count,)
            or offsets[0] != 0
            or offsets[-1] != len(indices)
            or np.any(offsets[1:] < offsets[:-1])
            or not np.array_equal(present, np.diff(offsets) > 0)
            or (len(indices) and (indices.min() < 0 or indices.max() >= count))
            or not np.isfinite(arrays["landmarks"]).all()
        ):
            raise ValueError(
                "FaceXray bank arrays have invalid shapes, references or values"
            )
        self._arrays, self._stats, self._pid = arrays, stats, os.getpid()

    def foreground(self, index):
        self._open()
        if not self._arrays["neighbor_present"][index]:
            return index, True
        start, end = self._arrays["neighbor_offsets"][index : index + 2]
        return int(random.choice(self._arrays["neighbor_indices"][start:end])), False

    def landmarks(self, index):
        self._open()
        return self._arrays["landmarks"][index]

    def receipt(self):
        self._open()
        return {
            "identity": self.identity,
            "manifest_sha256": self.header["manifest_sha256"],
            "origin": self.header["origin"],
            "selection_policy": self.header["selection_policy"],
            "arrays": self.header["arrays"],
            "members": len(self.header["member_ids"]),
            "neighbor_parents": int(self._arrays["neighbor_present"].sum()),
            "neighbor_edges": len(self._arrays["neighbor_indices"]),
        }
