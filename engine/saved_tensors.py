"""CPU saved-activation transport retaining tensor values/strides and model storage."""

import hashlib
import os
from pathlib import Path
import time
import weakref

import torch
from torch.multiprocessing.reductions import StorageWeakRef


def value_hash(tensor):
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def layout(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
    }


class DiskSavedTensors:
    """Temporary files are separate from scientific inputs/states/gradients.

    This audit keeps them until explicit successful-cleanup, and preserves them
    on failure. Each disk load validates full values and the original layout.
    No casts, recomputation, tensor detach in the mathematical graph, batch split
    or mode change is performed. Live source aliases receive version checks.
    """

    def __init__(self, model, directory, *, threshold=128 * 1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.persistent = {
            tensor.untyped_storage().data_ptr()
            for tensor in [*model.parameters(), *model.buffers()]
        }
        self.threshold = threshold
        self.cache = {}
        self.files = []
        self.loads = 0
        self.inline = 0
        self.views = 0
        self.last_advice = 0.0
        self.cache_advice_passes = 0

    def advise_cache(self):
        if time.monotonic() - self.last_advice < 3:
            return
        for entry in self.files:
            with Path(entry["path"]).open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.last_advice = time.monotonic()
        self.cache_advice_passes += 1

    def pack(self, tensor):
        if tensor.device.type != "cpu":
            raise ValueError("Audit disk transport is CPU-only")
        version = tensor._version
        if (
            tensor.untyped_storage().data_ptr() in self.persistent
            or tensor.untyped_storage().nbytes() < self.threshold
        ):
            self.inline += 1
            return {"tensor": tensor.detach(), "version": version}
        # C++ weak storage identity survives Python view-wrapper turnover and
        # cannot confuse a newly allocated storage at a reused data address.
        storage = StorageWeakRef(tensor.untyped_storage())
        key = storage.cdata, version, str(tensor.dtype)
        entry = self.cache.get(key)
        if entry is None:
            path = self.directory / f"{len(self.files):06d}.pth"
            torch.save(tensor.detach(), path)
            entry = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "storage_identity": storage,
                "views": 0,
            }
            self.files.append(entry)
            self.cache[key] = entry
            # File contents are already on disk; release clean cache pages
            # proactively instead of repeatedly pressing the memory ceiling.
            with path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        row = {
            "path": entry["path"],
            "layout": layout(tensor),
            "value_sha256": value_hash(tensor),
            "source": weakref.ref(tensor),
            "version": version,
        }
        base = tensor._base
        if base is not None:
            row["base"] = weakref.ref(base)
        entry["views"] += 1
        self.views += 1
        self.advise_cache()
        return row

    def unpack(self, row):
        if "tensor" in row:
            if row["tensor"]._version != row["version"]:
                raise RuntimeError("Native saved inline tensor was modified in-place")
            return row["tensor"]
        source = row["source"]()
        if source is None and "base" in row:
            source = row["base"]()
        if source is not None and source._version != row["version"]:
            raise RuntimeError("Native saved disk tensor was modified in-place")
        tensor = torch.load(
            row["path"], weights_only=True, mmap=True, map_location="cpu"
        )
        # A single full storage may represent several strided/sliced views.
        tensor = tensor.as_strided(
            row["layout"]["shape"],
            row["layout"]["stride"],
            storage_offset=row["layout"]["offset"],
        )
        assert (
            layout(tensor) == row["layout"]
            and value_hash(tensor) == row["value_sha256"]
        ), "Saved-tensor bytes/layout changed"
        self.loads += 1
        self.advise_cache()
        return tensor

    def __enter__(self):
        self.context = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self.context.__enter__()
        return self

    def __exit__(self, *error):
        return self.context.__exit__(*error)

    def report(self):
        return {
            "scope": "Byte-validated CPU autograd saved-tensor disk transport, not a mathematical model change",
            "files": [
                {key: value for key, value in row.items() if key != "storage_identity"}
                for row in self.files
            ],
            "disk_loads_verified": self.loads,
            "inline_saved": self.inline,
            "disk_views_saved": self.views,
            "cache_advice_passes": self.cache_advice_passes,
            "deduplication": "C++ weak storage identity, dtype and version; per-view exact layout/value checks",
            "bytes_written": sum(row["bytes"] for row in self.files),
            "batch_split": False,
            "dtype_change": False,
            "recomputation": False,
            "scratch_preserved": any(Path(row["path"]).exists() for row in self.files),
        }

    def cleanup(self):
        """Remove owned scratch only after successful backward without retain_graph.

        Preserve report() metadata separately and retain scratch on failures.
        Mathematical inputs, model/gradient/optimizer artifacts are not scratch.
        """
        for row in self.files:
            Path(row["path"]).unlink(missing_ok=True)
        self.directory.rmdir()
