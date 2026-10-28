"""Index Parquet metadata only, preserving shard/row-group/local-row identity."""

import pyarrow.parquet as pq

from data.manifest import Sample


def shard_paths(settings, split):
    patterns = settings.get("splits", {}).get(split)
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"Explicit Parquet patterns required for split {split}")
    files = set()
    for pattern in patterns:
        from data.manifest import relative_path

        relative_path(pattern)
        if "**" in pattern:
            raise ValueError("Recursive Parquet globs are not supported")
        matches = list(settings["root"].glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No Parquet files match {pattern}")
        files.update(matches)
    return sorted(files)


def build(settings, name, split, frame_count):
    if frame_count is not None:
        raise ValueError("frames_per_video applies to videos, not Parquet images")
    image_column = settings.get("image_column", "image")
    label_column = settings.get("label_column", "label")
    label_map = settings.get("label_map", {"real": 0, "fake": 1})
    metadata_columns = settings.get("metadata_columns", [])
    for path in shard_paths(settings, split):
        with pq.ParquetFile(path) as shard:
            required = {image_column, label_column, *metadata_columns}
            if not required.issubset(shard.schema_arrow.names):
                raise ValueError(
                    f"{path}: missing columns {required - set(shard.schema_arrow.names)}"
                )
            for group in range(shard.num_row_groups):
                rows = shard.read_row_group(
                    group, columns=[label_column, *metadata_columns]
                ).to_pylist()
                for index, row in enumerate(rows):
                    raw = str(row[label_column])
                    if raw not in label_map:
                        raise ValueError(f"{path}: unknown label {raw!r}")
                    relative = path.relative_to(settings["root"]).as_posix()
                    yield Sample(
                        id=f"{name}:{relative}:{group}:{index}",
                        dataset=name,
                        label=label_map[raw],
                        split=split,
                        storage="parquet",
                        path=relative,
                        row_group=group,
                        row_index=index,
                        metadata={key: row[key] for key in metadata_columns},
                    )
