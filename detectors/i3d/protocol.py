"""Source-domain isolation over entire frame pools, before clip exclusions."""

from pathlib import PurePosixPath


def validate_source_splits(train, validation):
    if train.header["split"] != "train" or validation.header["split"] != "val":
        raise ValueError("I3D training/selection require train/val source splits")
    if train.header["dataset"] != validation.header["dataset"]:
        raise ValueError("I3D checkpoint selection requires source-domain validation")
    if train.root.resolve() != validation.root.resolve():
        raise ValueError(
            "I3D source-domain manifests must resolve the same dataset root"
        )
    for dataset in (train, validation):
        if {clip.label for clip in dataset.records} != {0, 1}:
            raise ValueError("I3D selected training/validation clips need both classes")
        source = dataset.source_records
        if not source or any(
            row.dataset != dataset.header["dataset"]
            or row.split != dataset.header["split"]
            or not row.video_id
            for row in source
        ):
            raise ValueError(
                "I3D entire source pool needs consistent dataset/split/video identity"
            )
    source_train, source_val = train.source_records, validation.source_records
    identities = {
        "frame_id": lambda row: row.id,
        "image_path": lambda row: PurePosixPath(row.path).as_posix(),
        "video_id": lambda row: row.video_id,
    }
    for name, identity in identities.items():
        overlap = {identity(row) for row in source_train} & {
            identity(row) for row in source_val
        }
        if overlap:
            raise ValueError(
                f"I3D full-source train/validation {name} overlap: {sorted(overlap)[:5]}"
            )
    return {
        "policy": "Source-domain split isolation over all manifest rows, including short videos and truncated tails",
        "train_source_frames": len(source_train),
        "validation_source_frames": len(source_val),
        "train_selected_clips": len(train.records),
        "validation_selected_clips": len(validation.records),
        "checked_identities": list(identities),
        "current_filesystem_aliases_or_subject_identity_disjointness_verified": False,
    }
