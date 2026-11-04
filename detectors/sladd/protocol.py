"""Explicit precollected source inventory and native SLADD sidecar formats.

The formal source train_config sets lmdb=False. Its grayscale masks survive
pairDataset but fail the synthesizer's HWC3 indexing. The alternative transport
replays native lmdb=True per-key decoding on exported bytes; it is an explicit
configuration override, not a claim about the author training recipe or LMDB IO.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from data.manifest import read_manifest, relative_path, sha256


PROFILES = ("source_filesystem_gray", "source_lmdb_rgb_bytes")


def identity(root, relative, *, missing=False):
    relative_path(relative)
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("SLADD source path escapes the dataset root")
    if missing and not path.exists():
        return {"path": relative, "sha256": None, "bytes": None}
    return {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}


class SLADDSourcePlan:
    def __init__(self, base, *, profile):
        if profile not in PROFILES or base.header["split"] != "train":
            raise ValueError(
                "SLADD needs an explicit source format and train inventory"
            )
        self.profile = profile
        self.root = Path(base.root).resolve()
        self.manifest_path = Path(base.manifest_path).resolve()
        self.manifest_sha256 = sha256(self.manifest_path)
        header, rows = read_manifest(self.manifest_path)
        if header != base.header or rows != list(base.records):
            raise ValueError("SLADD must bind the complete unchanged parent manifest")
        self.header = deepcopy(header)
        self.source_records = tuple(
            replace(row, metadata=deepcopy(row.metadata)) for row in rows
        )
        self.bindings, self.specific_labels = {}, {}
        for row in self.source_records:
            if (
                row.storage != "image"
                or row.split != "train"
                or row.dataset != header["dataset"]
            ):
                raise ValueError("SLADD source adapter requires train image records")
            specific = row.metadata.get("specific_label", row.label)
            if (
                type(specific) is not int
                or specific < 0
                or (specific == 0) != (row.label == 0)
            ):
                raise ValueError(
                    "SLADD specific_label must be zero for real and positive for fake"
                )
            self.specific_labels[row.id] = specific
            image = identity(self.root, row.path)
            declared = header.get("sources", {}).get(row.path)
            if declared is not None and (
                declared["sha256"] != image["sha256"]
                or declared["size"] != image["bytes"]
            ):
                raise ValueError("SLADD image differs from its manifest identity")
            self.bindings[row.id] = {
                "image": image,
                "mask": self._sidecar(row, "mask"),
                "landmarks": self._sidecar(row, "landmarks"),
            }
        self.records = [row for row in self.source_records if row.label == 1]
        self.real_records = [row for row in self.source_records if row.label == 0]
        if not self.records or not self.real_records:
            raise ValueError(
                "SLADD requires indexed fake rows and a nonempty real pool"
            )

    def _sidecar(self, row, part):
        descriptor = row.metadata.get(part)
        if descriptor is None:
            if "frames" not in row.path or not row.path.endswith(".png"):
                raise ValueError(f"SLADD {row.id} needs an explicit {part} descriptor")
            relative = row.path.replace(
                "frames", "masks" if part == "mask" else "landmarks"
            )
            if part == "landmarks":
                relative = relative.replace(".png", ".npy")
            descriptor = {"path": relative}
            if part == "landmarks":
                descriptor["coordinate_size"] = 256
        if not isinstance(descriptor, dict):
            raise ValueError(f"SLADD {part} descriptor must be a mapping")
        required = {"path", "coordinate_size"} if part == "landmarks" else {"path"}
        if set(descriptor) not in (required, required | {"sha256"}):
            raise ValueError(f"Unsupported SLADD {part} descriptor")
        if part == "landmarks" and (
            type(descriptor["coordinate_size"]) is not int
            or descriptor["coordinate_size"] != 256
        ):
            raise ValueError(
                "SLADD landmarks require the declared source256 coordinate space"
            )
        pinned = identity(self.root, descriptor["path"], missing=True)
        if "sha256" in descriptor and descriptor["sha256"] != pinned["sha256"]:
            raise ValueError(f"SLADD bound {part} hash/absence differs from metadata")
        return pinned

    def verify_manifest(self):
        if sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("SLADD source manifest changed during loading")

    def file(self, source_id, part):
        pinned = self.bindings[source_id][part]
        current = identity(self.root, pinned["path"], missing=part != "image")
        if current != pinned:
            raise ValueError(f"SLADD bound {part} changed: {source_id}")
        return None if current["sha256"] is None else self.root / current["path"]

    def receipt(self):
        return {
            "profile": self.profile,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "indexed_fake_ids": [row.id for row in self.records],
            "real_pool_ids": [row.id for row in self.real_records],
            "bindings": deepcopy(self.bindings),
            "specific_labels": dict(self.specific_labels),
            "inventory_policy": "Complete precollected manifest order;no class/path deduplication,frame reselection or upstream JSON discovery/shuffle reproduction",
            "sampling_policy": "Fake indexed;Python randint real sampling with replacement before IO;decode and augment fake before real;collate real-all then fake-all",
            "source_filesystem_gray": "Formal source lmdb=False:PNG gray masks and NPY landmarks;native grayscale synthesis failure is retained",
            "source_lmdb_rgb_bytes": "Explicit source lmdb=True override:exported encoded image/mask bytes and rawuint32(81,2) landmark bytes;not an LMDB database reader or author-recipe claim",
            "pair_batch_unit": "Native32 pairs ->64 collated views;policy32 ->64 generated views",
        }
