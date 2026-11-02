"""Mapped ordered neighbors; coordinates stay in separately pinned NPY files."""

import json
import os
from pathlib import Path
import random

import numpy as np

from data.manifest import relative_path, sha256


ARRAYS = {"offsets": np.dtype("int64"), "indices": np.dtype("int64")}


def bank_identity(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_bank(output, *, manifest_path, records, neighbors, origin):
    """Present lists retain order/duplicates; absent entries use self without RNG."""
    count = len(records)
    if not count or any(type(i) is not int or not 0 <= i < count for i in neighbors):
        raise ValueError("PCL neighbor parent outside training inventory")
    offsets, indices = [0], []
    for index in range(count):
        values = neighbors.get(index, [])
        if index in neighbors and not values:
            raise ValueError("PCL present empty neighbors are not self fallback")
        if any(type(i) is not int or not 0 <= i < count for i in values):
            raise ValueError("PCL foreground outside complete training inventory")
        indices.extend(values)
        offsets.append(len(indices))
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for key, values in (("offsets", offsets), ("indices", indices)):
        path = output / f"{key}.npy"
        np.save(path, np.asarray(values, dtype=ARRAYS[key]), allow_pickle=False)
        arrays[key] = {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    path = output / "bank.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "recipe": "i2g_ordered_neighbors_v1",
                "manifest_sha256": sha256(Path(manifest_path)),
                "member_ids": [r.id for r in records],
                "arrays": arrays,
                "origin": origin,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return bank_identity(path)


class I2GBank:
    def __init__(self, identity, dataset):
        if not isinstance(identity, dict) or set(identity) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError("PCL bank requires path/hash/bytes identity")
        self.identity = dict(identity)
        self._pid, self._arrays, self._stats = None, {}, {}
        if bank_identity(identity["path"]) != identity:
            raise ValueError("PCL neighbor bank changed")
        self.header = json.loads(Path(identity["path"]).read_text())
        if (
            self.header.get("version") != 1
            or self.header.get("recipe") != "i2g_ordered_neighbors_v1"
            or set(self.header.get("arrays", {})) != set(ARRAYS)
            or dataset.header["split"] != "train"
            or self.header.get("manifest_sha256") != sha256(dataset.manifest_path)
            or self.header.get("member_ids") != [r.id for r in dataset.records]
        ):
            raise ValueError("PCL bank must bind complete exact train manifest")
        self._open()

    def __getstate__(self):
        return {**self.__dict__, "_pid": None, "_arrays": {}, "_stats": {}}

    def _open(self):
        if self._pid == os.getpid():
            for name, expected in self._stats.items():
                info = Path(name).stat()
                if (info.st_size, info.st_mtime_ns) != expected:
                    raise ValueError("PCL mapped neighbor bank changed during training")
            return
        if bank_identity(self.identity["path"]) != self.identity:
            raise ValueError("PCL neighbor bank changed before worker loading")
        root = Path(self.identity["path"]).parent
        paths = [Path(self.identity["path"])]
        arrays = {}
        for key, dtype in ARRAYS.items():
            row = self.header["arrays"][key]
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
                raise ValueError("PCL neighbor array requires path/hash/bytes")
            path = root / relative_path(row["path"])
            if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("PCL neighbor array changed")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != dtype or array.ndim != 1:
                raise ValueError("PCL neighbor array dtype/shape changed")
            arrays[key] = array
            paths.append(path)
        offsets, indices = arrays["offsets"], arrays["indices"]
        if (
            offsets.shape != (len(self.header["member_ids"]) + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(indices)
            or np.any(offsets[1:] < offsets[:-1])
            or (
                len(indices)
                and (indices.min() < 0 or indices.max() >= len(offsets) - 1)
            )
        ):
            raise ValueError("PCL neighbor offsets/references invalid")
        self._stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
        self._arrays, self._pid = arrays, os.getpid()

    def foreground(self, parent):
        self._open()
        start, end = self._arrays["offsets"][parent : parent + 2]
        if start == end:
            return parent, True
        return int(random.choice(self._arrays["indices"][start:end])), False

    def receipt(self):
        self._open()
        return {
            "identity": self.identity,
            "header": self.header,
            "neighbor_edges": len(self._arrays["indices"]),
        }
