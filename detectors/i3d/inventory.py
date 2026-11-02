"""Bind complete frame-manifest snapshots without scanning source directories."""

from collections import OrderedDict
from pathlib import Path, PurePosixPath

from data.manifest import read_manifest, read_metadata, sha256
from .clips import VideoFrames


class I3DInventory:
    """Completeness is relative to the declared preparation snapshot.

    A bound files-first metadata snapshot and an unlimited/all-frame sampling
    declaration are required. This does not establish current filesystem or
    original-author corpus completeness. No new data discovery takes place.
    """

    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.header, records = read_manifest(self.manifest_path)
        sampling = self.header.get("sampling", {})
        if (
            set(sampling) != {"frames_per_video", "algorithm", "limit"}
            or sampling["frames_per_video"] is not None
            or sampling["limit"] is not None
            or sampling["algorithm"] != "uniform_endpoints_v1"
        ):
            raise ValueError("I3D requires an unlimited all-frame source manifest")
        if self.header.get("adapter") not in {
            "ffpp",
            "celeb_df",
            "directories",
            "dfdc",
        }:
            raise ValueError("I3D needs an explicit numeric-frame image inventory")
        metadata = read_metadata(self.manifest_path, self.header)
        if metadata is None or metadata.get("policy") != "files_first":
            raise ValueError("I3D requires bound files-first source metadata")
        videos = metadata.get("available_video_metadata")
        if not isinstance(videos, dict) or not videos:
            raise ValueError(
                "I3D source metadata must declare its available video pools"
            )
        grouped = OrderedDict()
        for record in records:
            declared = videos.get(record.video_id)
            if (
                record.storage != "image"
                or not isinstance(declared, dict)
                or type(declared.get("label")) is not int
                or declared["label"] != record.label
                or declared.get("category") != record.category
                or declared.get("frames")
                != PurePosixPath(record.path).parent.as_posix()
                or not PurePosixPath(record.path).stem.isdigit()
                or record.frame_index != int(PurePosixPath(record.path).stem)
            ):
                raise ValueError(
                    f"I3D source row differs from its video inventory: {record.id}"
                )
            grouped.setdefault(record.video_id, []).append(record)
        if (
            set(grouped) != set(videos)
            or type(metadata.get("available_video_count")) is not int
            or metadata["available_video_count"] != len(grouped)
        ):
            raise ValueError("I3D manifest/metadata video coverage differs")
        self.source_records = tuple(records)
        self.pools = tuple(
            VideoFrames(
                self.header["dataset"],
                self.header["split"],
                video_id,
                tuple(frames),
                complete_pool=True,
            )
            for video_id, frames in grouped.items()
        )
        metadata_path = (
            self.manifest_path.parent / self.header["reconciliation"]["metadata_file"]
        )
        self.provenance = {
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256(self.manifest_path),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "scope": "All rows of the bound declared all-frame preparation snapshot",
            "source_frames": len(records),
            "source_videos": len(self.pools),
            "video_order": "First video appearance in source manifest; not a claim of original-author metadata order",
            "fresh_disk_inventory_verified": False,
            "original_author_corpus_completeness_verified": False,
            "excluded_source_rows_retained_for_validation": True,
        }
