"""Explicit native SBIPlusV2 pairing and ordered-neighbor source bindings.

Windows source strings describe native grouping/sorting; actual files use the
remake manifest's relative POSIX paths. No image generation or source discovery
occurs here. The declared manifest order replaces upstream dataset discovery.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from data.manifest import read_manifest, relative_path, sha256, write_manifest


RECIPE = "prodet_native_sbiplus_v2_raw_neighbors_v1"


def neighbor_identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


class ProDetSourcePreparer:
    """Bind train8 images/landmarks and build train-only landmark neighbors.

    Uses DeepfakeBench's flattened-float32 landmark KDTree / ordered nearest100
    recipe. The original all-split pickle is deliberately not used: every BI
    foreground must belong to the same frozen train8 manifest. This is recorded
    as a source-isolation adaptation, not an identical author pickle replay.
    """

    def prepare(self, dataset, output):
        import numpy as np
        from scipy.spatial import KDTree
        from engine.protocol import VideoSamplingPolicy

        if dataset.header["dataset"] != "FaceForensics++" or dataset.header["split"] != "train":
            raise ValueError("ProDet preparation requires the FF++ train8 manifest")
        sampling = VideoSamplingPolicy({"name": "train8_test32", "train_frames": 8,
                                        "test_frames": 32}).validate(dataset.header, dataset.records)
        if dataset.header["settings"].get("compression") != "c23":
            raise ValueError("Author ProDet preparation requires FF++ c23")
        output = Path(output).expanduser().absolute()
        neighbors_path = output.with_suffix(".neighbors.json")
        receipt_path = output.with_suffix(".preparation.json")
        for path in (output, neighbors_path, receipt_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"ProDet preparation requires new files: {path}")
        records, arrays, keys, sources = [], [], [], deepcopy(dataset.header["sources"])
        shapes = set()
        for row in dataset.records:
            if row.storage != "image" or not row.path.endswith(".png"):
                raise ValueError("Author ProDet requires filesystem PNG image sources")
            image = file_identity(dataset.root, row.path)
            sources[row.path] = {"sha256": image["sha256"], "size": image["bytes"]}
            native = "prodet:\\" + row.path.replace("/", "\\")
            attributes = {**row.metadata, "prodet_native_path": native}
            if row.label == 0:
                parts = list(Path(row.path).parts)
                if parts.count("frames") != 1:
                    raise ValueError("ProDet real images require one frames path component")
                parts[parts.index("frames")] = "landmarks"
                relative = Path(*parts).with_suffix(".npy").as_posix()
                identity = file_identity(dataset.root, relative)
                values = np.load(Path(dataset.root) / relative, allow_pickle=False)
                if (values.shape not in ((68, 2), (81, 2)) or values.dtype.kind not in "iuf"
                        or not np.isfinite(values).all()):
                    raise ValueError(f"Invalid ProDet landmarks: {relative}")
                shapes.add(values.shape)
                # Both generation and the upstream nearest script use the native
                # 256-coordinate sidecar. Never rescale or synthesize zero points.
                attributes["landmarks"] = {"path": relative, "sha256": identity["sha256"],
                                             "coordinate_size": 256}
                arrays.append(values.astype(np.float32).ravel())
                keys.append(native.replace("frames", "landmarks").replace(".png", ".npy"))
            records.append(replace(row, metadata=attributes))
        if len(shapes) != 1 or len(arrays) < 2:
            raise ValueError("ProDet neighbors need at least two real sources with equal landmark shapes")
        vectors = np.asarray(arrays)
        tree = KDTree(vectors)
        raw = {}
        for key, vector in zip(keys, vectors, strict=True):
            _, indices = tree.query(vector, k=min(101, len(keys)))
            raw[key] = [keys[int(index)] for index in indices[1:]]
        if any(not values for values in native_neighbor_filter(raw).values()):
            raise ValueError("ProDet nearest table has empty entries after author filtering")
        header = {**dataset.header, "sources": sources,
                  "prodet_preparation": {"version": 1,
                    "parent_manifest": str(dataset.manifest_path),
                    "parent_manifest_sha256": sha256(dataset.manifest_path),
                    "sampling": sampling,
                    "landmark_coordinate_size": 256,
                    "neighbors": "train-only KDTree float32 flattened nearest100;author ordered filtering at load"}}
        # The files-first inventory belongs to the original manifest bytes.
        # Preserve that reference as parent provenance instead of pretending
        # its bound-manifest SHA/header also describe this derived manifest.
        if "reconciliation" in header:
            header["prodet_preparation"]["parent_reconciliation"] = header.pop("reconciliation")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(output, header, records)
        header, records = read_manifest(output)
        bound = SimpleNamespace(root=dataset.root, manifest_path=output, header=header, records=records)
        identity = write_neighbors(neighbors_path, dataset=bound, neighbors=raw, origin={
            "algorithm_source": "https://github.com/SCLBD/DeepfakeBench/blob/main/training/dataset/generate_xray_nearest.py",
            "scope": "Only real members of the exact train8 manifest;no validation/test or extra frames",
            "author_original_pickle_reproduced": False,
            "numpy_version": np.__version__,
            "scipy_version": __import__("scipy").__version__,
            "neighbor_count": min(100, len(keys) - 1),
            "landmark_shape": list(next(iter(shapes))),
        })
        plan = ProDetSourcePlan(bound, neighbors=identity)
        receipt = {"status": "prepared", "manifest": str(output),
                   "manifest_sha256": sha256(output), "samples": len(records),
                   "training_neighbors": identity, "sampling": sampling,
                   "quadruples": len(plan.pairs), "model_views": len(plan.records),
                   "source_protocol": header["prodet_preparation"],
                   "GPU_forward_executed": False}
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return receipt


def native_group(path):
    if not isinstance(path, str) or len(path.split("\\")) < 2:
        raise ValueError(
            "ProDet grouping requires an explicit native Windows source path"
        )
    video = path.split("\\")[-2]
    if not video or not path.endswith(".png"):
        raise ValueError(
            "ProDet native source paths require a video folder and PNG frame"
        )
    return video[:3] if len(video) > 4 else video


def native_neighbor_filter(neighbors):
    """Preserve init_nearest's deletion-during-iteration, including residuals."""
    if not isinstance(neighbors, dict) or any(
        not isinstance(key, str)
        or not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        for key, values in neighbors.items()
    ):
        raise ValueError("ProDet raw neighbors require ordered string lists")
    result = deepcopy(neighbors)
    for key, values in result.items():
        for value in values:
            if key[:-7] == value[:-7]:
                result[key].remove(value)
    return result


def write_neighbors(output, *, dataset, neighbors, origin):
    """Compile a declared RAW table; never invent/reorder/filter its neighbors.

    Native pickle loading, if used, belongs in an explicit trusted conversion
    command which retains its original file and records its hash in origin.
    """
    native_neighbor_filter(neighbors)  # Validate without altering the saved raw table.
    output = Path(output).expanduser().resolve()
    payload = {
        "version": 1,
        "recipe": RECIPE,
        "manifest_sha256": sha256(dataset.manifest_path),
        "member_ids": [row.id for row in dataset.records],
        "neighbors": neighbors,
        "origin": origin,
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(encoded)
    return {
        "path": str(output),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
    }


def file_identity(root, relative):
    relative_path(relative)
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("ProDet source/landmark path escapes the dataset root")
    return {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}


def verify_file(root, identity):
    current = file_identity(root, identity["path"])
    if current != identity:
        raise ValueError(f"ProDet bound source/landmark changed: {identity['path']}")
    return Path(root) / identity["path"]


class ProDetSourcePlan:
    """Selected train8/real_choice8/domain4 recipe over a complete manifest."""

    def __init__(self, dataset, *, neighbors):
        if dataset.header["split"] != "train":
            raise ValueError("ProDet specialized source plans are train-only")
        self.root = Path(dataset.root).resolve()
        self.manifest_path = Path(dataset.manifest_path).resolve()
        self.manifest_sha256 = sha256(self.manifest_path)
        manifest_header, manifest_records = read_manifest(self.manifest_path)
        if manifest_header != dataset.header or manifest_records != list(
            dataset.records
        ):
            raise ValueError(
                "ProDet source plan must use the unfiltered exact parent manifest"
            )
        self.header = deepcopy(dataset.header)
        self.source_records = tuple(
            replace(row, metadata=deepcopy(row.metadata)) for row in dataset.records
        )
        self.by_id, self.by_landmark, self.bindings = {}, {}, {}
        real, fake, native_paths = {}, {}, set()
        for row in self.source_records:
            native = row.metadata.get("prodet_native_path")
            group = native_group(native)
            if (
                row.id in self.by_id
                or native in native_paths
                or row.storage != "image"
                or row.dataset != self.header["dataset"]
                or row.split != "train"
                or not row.video_id
            ):
                raise ValueError(
                    "ProDet requires unique complete train image/native/video identities"
                )
            image = file_identity(self.root, row.path)
            expected = self.header.get("sources", {}).get(row.path)
            if expected is not None and (
                image["sha256"] != expected["sha256"]
                or image["bytes"] != expected["size"]
            ):
                raise ValueError("ProDet image differs from manifest source hash")
            binding = {"image": image, "native_path": native, "group": group}
            if row.label == 0:
                descriptor = row.metadata.get("landmarks")
                if (
                    not isinstance(descriptor, dict)
                    or set(descriptor) != {"path", "sha256", "coordinate_size"}
                    or type(descriptor["coordinate_size"]) is not int
                    or descriptor["coordinate_size"] != 256
                ):
                    raise ValueError(
                        "ProDet real sources require explicit 256-coordinate landmark path/hash"
                    )
                points = file_identity(self.root, descriptor["path"])
                if points["sha256"] != descriptor["sha256"]:
                    raise ValueError("ProDet landmark differs from its declared hash")
                key = native.replace("frames", "landmarks").replace(".png", ".npy")
                if key in self.by_landmark:
                    raise ValueError("ProDet native landmark identities are ambiguous")
                self.by_landmark[key] = row.id
                binding.update(landmarks=points, native_landmark_path=key)
            (real if row.label == 0 else fake).setdefault(group, []).append(row)
            self.by_id[row.id], self.bindings[row.id] = row, binding
            native_paths.add(native)
        for groups in (real, fake):
            for rows in groups.values():
                rows.sort(
                    key=lambda row: (row.metadata["prodet_native_path"], row.label)
                )
        if "281" not in fake:
            raise ValueError(
                "Native SBIPlusV2 requires fake group281 before its mandatory exclusion"
            )
        self.excluded_fake_ids = [row.id for row in fake.pop("281")]
        if not fake:
            raise ValueError(
                "ProDet requires a nonempty fake group after native281 exclusion"
            )
        self.groups = list(fake)
        self.pairs, self.records = [], []
        for group, rows in fake.items():
            if group not in real or not real[group]:
                raise ValueError(
                    f"ProDet fake group has no paired real sources: {group}"
                )
            for slot in range(32):
                requested_real = slot % 8
                source_real, source_fake = (
                    real[group][min(requested_real, len(real[group]) - 1)],
                    rows[min(slot, len(rows) - 1)],
                )
                pair = {
                    "id": f"prodet:{group}:{slot:02d}",
                    "group": group,
                    "slot": slot,
                    "requested_real_slot": requested_real,
                    "real_id": source_real.id,
                    "fake_id": source_fake.id,
                    "real_clamped": requested_real >= len(real[group]),
                    "fake_clamped": slot >= len(rows),
                }
                self.pairs.append(pair)
                for domain, label, parent in (
                    ("real", 0, source_real),
                    ("blendfake", 1, source_real),
                    ("bifake", 1, source_real),
                    ("deepfake", 1, source_fake),
                ):
                    self.records.append(
                        replace(
                            parent,
                            id=f"{pair['id']}:{domain}",
                            label=label,
                            metadata={
                                **parent.metadata,
                                "prodet_pair": pair.copy(),
                                "prodet_view": domain,
                            },
                        )
                    )
        if not isinstance(neighbors, dict) or set(neighbors) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError("ProDet needs the complete compiled raw-neighbor identity")
        self.neighbor_identity = dict(neighbors)
        path = Path(neighbors["path"])
        if (
            not path.is_absolute()
            or sha256(path) != neighbors["sha256"]
            or path.stat().st_size != neighbors["bytes"]
        ):
            raise ValueError("ProDet raw-neighbor file identity changed")
        self.neighbor_header = json.loads(path.read_text())
        header = self.neighbor_header
        if (
            header.get("version") != 1
            or header.get("recipe") != RECIPE
            or header.get("manifest_sha256") != self.manifest_sha256
            or header.get("member_ids") != [row.id for row in self.source_records]
        ):
            raise ValueError(
                "ProDet raw-neighbor table must bind the exact complete train manifest"
            )
        raw = header["neighbors"]
        self.neighbors = native_neighbor_filter(raw)
        if any(
            key not in self.by_landmark
            or any(value not in self.by_landmark for value in values)
            for key, values in raw.items()
        ):
            raise ValueError(
                "ProDet raw neighbor references a non-real or external/non-training source"
            )
        if any(not values for values in self.neighbors.values()):
            raise ValueError(
                "ProDet neighbor entry is empty after native filter;not a self fallback"
            )

    def verify_control_files(self):
        if sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("ProDet parent manifest changed during use")
        path = Path(self.neighbor_identity["path"])
        if (
            path.stat().st_size != self.neighbor_identity["bytes"]
            or sha256(path) != self.neighbor_identity["sha256"]
        ):
            raise ValueError("ProDet raw-neighbor file changed during use")

    def selection_receipt(self):
        self.verify_control_files()
        same_video_residuals = sum(
            key[:-7] == value[:-7]
            for key, values in self.neighbors.items()
            for value in values
        )
        return {
            "recipe": RECIPE,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "source_pool_count": len(self.source_records),
            "quadruples": len(self.pairs),
            "view_count": len(self.records),
            "group_order": self.groups,
            "excluded_fake_group281_ids": self.excluded_fake_ids,
            "native_short_list_clamped_quadruples": sum(
                pair["real_clamped"] or pair["fake_clamped"] for pair in self.pairs
            ),
            "bindings": self.bindings,
            "pairs": self.pairs,
            "neighbor_identity": self.neighbor_identity,
            "neighbor_origin": self.neighbor_header.get("origin"),
            "raw_neighbor_edges": sum(
                map(len, self.neighbor_header["neighbors"].values())
            ),
            "filtered_neighbor_edges": sum(map(len, self.neighbors.values())),
            "native_same_video_neighbor_residuals": same_video_residuals,
            "filtered_neighbors_sha256": hashlib.sha256(
                json.dumps(self.neighbors, sort_keys=True).encode()
            ).hexdigest(),
            "source_binding_policy": "Complete declared ordered train manifest, including281/excluded and unused sources;all raw neighbor references,including tails,must resolve to real train members. No hidden filtering,nearest regeneration orreplacement.",
            "source_repairs": [
                "Explicit POSIX file bindings plus native Windows identity strings replace implicitdrive/path discovery;upstream corpus discovery/RNG is not reproduced.",
                "Compiled hashed raw JSON table replaces implicitpickle reads;native mutation-while-iterating filter is retained.",
                "Required finite landmark sidecars replace native missing-file zero arrays at future decode;no coordinate rescale.",
                "Invalid bindings fail before sampling;native unused landmark_dict pickle loading is not reproduced.",
            ],
            "full_dataset_or_subject_identity_disjointness_verified": False,
        }


def validate_source_splits(train_plan, validation):
    """Check all original sources, including excluded groups and neighbor tails."""
    if (
        validation.header["split"] != "val"
        or validation.header["dataset"] != train_plan.header["dataset"]
        or Path(validation.root).resolve() != train_plan.root
    ):
        raise ValueError(
            "ProDet validation must use the same source domain/root and val split"
        )
    train, val = train_plan.source_records, validation.records
    if not val or any(
        row.split != "val"
        or row.dataset != train_plan.header["dataset"]
        or not row.video_id
        for row in val
    ):
        raise ValueError("ProDet validation source pool lacks consistent identities")
    for name, key in (
        ("id", lambda row: row.id),
        ("path", lambda row: row.path),
        ("video", lambda row: row.video_id),
    ):
        overlap = {key(row) for row in train} & {key(row) for row in val}
        if overlap:
            raise ValueError(f"ProDet complete-source train/validation {name} overlap")
    return {
        "train_source_count": len(train),
        "validation_source_count": len(val),
        "checked_identities": ["id", "path", "video"],
        "includes_excluded281_and_all_neighbor_members": True,
        "subject_or_filesystem_alias_disjointness_verified": False,
    }
