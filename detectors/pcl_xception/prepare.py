"""Offline import of caller-selected trusted native nearest-face lists."""

import argparse
import json
from pathlib import Path
import pickle

from data.manifest import read_manifest, sha256
from .bank import write_bank


def prepare_bank(*, manifest, neighbors_path, output):
    manifest, neighbors_path = Path(manifest).resolve(), Path(neighbors_path).resolve()
    header, records = read_manifest(manifest)
    if header["split"] != "train":
        raise ValueError("PCL neighbor preparation requires train manifest")
    # Unpickle only the explicitly supplied trusted asset, never during fit/load.
    with neighbors_path.open("rb") as stream:
        native = pickle.load(stream)
    keys, lookup = [], {}
    for index, record in enumerate(records):
        key = record.metadata.get("pcl_bank_id")
        if not isinstance(key, str) or not key:
            raise ValueError(
                "Every PCL train member needs explicit pcl_bank_id binding"
            )
        if key in lookup:
            prior = records[lookup[key]]
            if (
                record.storage,
                record.path,
                record.row_group,
                record.row_index,
                record.label,
            ) != (
                prior.storage,
                prior.path,
                prior.row_group,
                prior.row_index,
                prior.label,
            ):
                raise ValueError(
                    "PCL duplicate bank binding refers to different sources"
                )
        else:
            lookup[key] = index
        keys.append(key)
    neighbors = {}
    for index, key in enumerate(keys):
        if key not in native:
            continue
        values = native[key]
        if not isinstance(values, list) or not values:
            raise ValueError("PCL present native neighbors must be a nonempty list")
        if any(value not in lookup for value in values):
            raise ValueError(
                "PCL native foreground outside train inventory; supply complete bindings, no silent filtering"
            )
        neighbors[index] = [lookup[value] for value in values]
    return write_bank(
        output,
        manifest_path=manifest,
        records=records,
        neighbors=neighbors,
        origin={
            "neighbors": {
                "path": str(neighbors_path),
                "sha256": sha256(neighbors_path),
            },
            "native_neighbor_parents": len(native),
            "provided_lists_order_duplicates_preserved": True,
            "coordinate_source": "Per-frame pinned NPY, never unused landmark_dict_ffall.pkl",
            "background_order": "Stable first real occurrence in manifest replaces native list(set(...)) hash order",
            "full_original_protocol_reproduced": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--neighbors-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_bank(
                manifest=args.manifest,
                neighbors_path=args.neighbors_pickle,
                output=args.output,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
