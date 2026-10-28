"""Versioned JSONL manifests: header first, then one record per image."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value == ".":
        raise ValueError(f"Expected a relative path without traversal: {value}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


INVENTORY_FIELDS = (
    "policy",
    "reference_video_count",
    "available_video_count",
    "stale_metadata",
    "files_outside_requested_metadata",
    "available_video_metadata",
)


def inventory_digest(metadata: dict) -> str:
    payload = {key: metadata[key] for key in INVENTORY_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Sample:
    id: str
    dataset: str
    label: int
    split: str
    storage: str
    path: str
    video_id: str = ""
    frame_index: int | None = None
    category: str = ""
    row_group: int | None = None
    row_index: int | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        relative_path(self.path)
        if not all(
            isinstance(v, str) and v for v in (self.id, self.dataset, self.split)
        ):
            raise ValueError("id, dataset and split must be nonempty strings")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("Labels must be integers: real=0, fake=1")
        if self.storage not in ("image", "parquet"):
            raise ValueError(f"Unknown storage: {self.storage}")
        if self.storage == "parquet" and any(
            type(v) is not int or v < 0 for v in (self.row_group, self.row_index)
        ):
            raise ValueError(
                "Parquet records require nonnegative row_group and row_index"
            )
        if self.frame_index is not None and (
            type(self.frame_index) is not int or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer")


def read_manifest(path: Path | str) -> tuple[dict, list[Sample]]:
    with Path(path).open() as stream:
        header = json.loads(next(stream))["manifest"]
        if header.get("version") != 1:
            raise ValueError("Unsupported manifest version")
        records, seen = [], set()
        for line_number, line in enumerate(stream, 2):
            try:
                record = Sample(**json.loads(line))
                if record.id in seen:
                    raise ValueError(f"Duplicate sample ID: {record.id}")
                if (
                    record.dataset != header["dataset"]
                    or record.split != header["split"]
                ):
                    raise ValueError(
                        "Record dataset/split differs from manifest header"
                    )
                seen.add(record.id)
                records.append(record)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return header, records


def read_metadata(path: Path | str, header: dict | None = None) -> dict | None:
    """Read only the derived metadata bound to this exact manifest snapshot."""
    path = Path(path)
    if header is None:
        with path.open() as stream:
            header = json.loads(next(stream))["manifest"]
    name = header.get("reconciliation", {}).get("metadata_file")
    if name is None:
        return None
    relative_path(name)
    if len(PurePosixPath(name).parts) != 1:
        raise ValueError("Derived metadata must be a sibling of the manifest")
    metadata = json.loads((path.parent / name).read_text())
    if (
        metadata.get("manifest_sha256") != sha256(path)
        or metadata.get("manifest_header") != header
        or inventory_digest(metadata)
        != header["reconciliation"].get("inventory_sha256")
    ):
        raise ValueError(f"Metadata does not match manifest snapshot: {path}")
    return metadata


def write_manifest(path: Path, header: dict, records: Iterable[Sample]) -> int:
    """Replace atomically only after the entire input has been validated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(
                json.dumps({"manifest": {"version": 1, **header}}, sort_keys=True)
                + "\n"
            )
            seen = set()
            for record in records:
                if record.id in seen:
                    raise ValueError(f"Duplicate sample ID: {record.id}")
                if (
                    record.dataset != header["dataset"]
                    or record.split != header["split"]
                ):
                    raise ValueError(
                        "Record dataset/split differs from manifest header"
                    )
                seen.add(record.id)
                stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            if not seen:
                raise ValueError(
                    "No samples found; refusing to write an empty manifest"
                )
        os.replace(temporary, path)
        return len(seen)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    """Atomically publish derived metadata without altering upstream files."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
