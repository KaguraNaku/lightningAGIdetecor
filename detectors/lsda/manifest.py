"""Derive native-order LSDA group manifests from explicit FF++ split pairs."""

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import tempfile

from data.manifest import (
    read_manifest,
    read_metadata,
    sha256,
    write_json,
    write_manifest,
)


DEFAULT_CATEGORY_MAP = {
    "youtube": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
}


class LSDAInventoryBuilder:
    """Bind existing rows; never decode, resample, relabel or silently drop a group."""

    def __init__(self, category_map=None):
        if category_map is not None and not isinstance(category_map, dict):
            raise ValueError("LSDA category map must be a dictionary")
        self.category_map = dict(
            DEFAULT_CATEGORY_MAP if category_map is None else category_map
        )
        if (
            len(self.category_map) != 5
            or any(
                not isinstance(key, str) or not key or type(value) is not int
                for key, value in self.category_map.items()
            )
            or set(self.category_map.values()) != set(range(5))
        ):
            raise ValueError(
                "LSDA grouping requires exactly five explicit category names mapped to0..4"
            )

    def bind(self, records, pairs, *, source_sha256):
        if (
            not isinstance(pairs, list)
            or not pairs
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    not isinstance(value, str) or not value.isdigit() for value in pair
                )
                or pair[0] == pair[1]
                for pair in pairs
            )
        ):
            raise ValueError(
                "LSDA pairs must be nonempty distinct numeric-string FF++ identity pairs"
            )
        videos, ids = defaultdict(list), set()
        for row in records:
            if row.id in ids:
                raise ValueError("LSDA source manifest has duplicate row ids")
            ids.add(row.id)
            if row.category not in self.category_map:
                continue
            domain = self.category_map[row.category]
            key = row.video_id.removeprefix(row.category + "/")
            pattern = r"\d+" if domain == 0 else r"\d+_\d+"
            if (
                not re.fullmatch(pattern, key)
                or (domain == 0) != (row.label == 0)
                or row.split != "train"
            ):
                raise ValueError(
                    f"LSDA source {row.id} has ambiguous FF++ video/domain identity"
                )
            if "specific_label" in row.metadata and (
                type(row.metadata["specific_label"]) is not int
                or row.metadata["specific_label"] != domain
            ):
                raise ValueError(
                    f"LSDA source {row.id} conflicts with its explicit category map"
                )
            videos[domain, key].append(row)
        groups, bound, used, directed = [], [], set(), set()
        for pair in pairs:
            for first, second in (pair, pair[::-1]):
                if (first, second) in directed:
                    raise ValueError("LSDA split contains a duplicated directed pair")
                directed.add((first, second))
                group, group_sources = len(groups), []
                for domain in range(5):
                    key = first if domain == 0 else f"{first}_{second}"
                    selected = videos.get((domain, key), [])
                    if len(selected) != 32:
                        raise ValueError(
                            f"LSDA group{group} domain{domain} video{key} needs exactly32 existing source rows, got{len(selected)}"
                        )
                    if len({row.video_id for row in selected}) != 1:
                        raise ValueError("LSDA source video binding is ambiguous")
                    for slot, row in enumerate(selected):
                        bound.append(
                            replace(
                                row,
                                id=f"lsda:g{group}:{row.id}",
                                metadata={
                                    **row.metadata,
                                    "specific_label": domain,
                                    "lsda_group": group,
                                    "lsda_frame_slot": slot,
                                    "lsda_source_id": row.id,
                                    "lsda_source_manifest_sha256": source_sha256,
                                },
                            )
                        )
                        group_sources.append(row.id)
                        used.add(row.id)
                groups.append(
                    {
                        "group": group,
                        "pair": [first, second],
                        "source_ids": group_sources,
                    }
                )
        return bound, {
            "groups": groups,
            "input_rows": len(records),
            "output_rows": len(bound),
            "unique_source_rows_used": len(used),
            "unused_source_ids": [row.id for row in records if row.id not in used],
            "category_map": self.category_map,
            "slot_policy": "Preserve the32 existing rows per video in declared source-manifest order; no new image selection",
        }

    def build(self, source, pairs_path, output):
        source, pairs_path, output = (
            Path(value).expanduser().resolve() for value in (source, pairs_path, output)
        )
        protocol = output.with_suffix(".groups.json")
        if output.exists() or protocol.exists() or output == source:
            raise FileExistsError(
                "LSDA grouping only writes fresh manifest/protocol paths"
            )
        header, records = read_manifest(source)
        upstream_metadata = read_metadata(source, header)
        if (
            header["split"] != "train"
            or header.get("sampling", {}).get("limit") is not None
        ):
            raise ValueError("LSDA grouping requires an unlimited train manifest")
        source_digest, pairs_digest = sha256(source), sha256(pairs_path)
        bound, receipt = self.bind(
            records, json.loads(pairs_path.read_text()), source_sha256=source_digest
        )
        derivation = {
            "profile": "lsda-explicit-FFpp-groups-v1",
            "source_manifest": str(source),
            "source_manifest_sha256": source_digest,
            "pairs": str(pairs_path),
            "pairs_sha256": pairs_digest,
            "category_map": self.category_map,
            "frame_slots": 32,
            "groups": len(receipt["groups"]),
            "protocol_file": protocol.name,
        }
        if upstream_metadata is not None:
            metadata_path = source.parent / header["reconciliation"]["metadata_file"]
            derivation["upstream_metadata"] = {
                "path": str(metadata_path),
                "sha256": sha256(metadata_path),
                "inventory_sha256": header["reconciliation"]["inventory_sha256"],
            }
        derived = deepcopy(header)
        # The original reconciliation sidecar is tied to the original manifest.
        # Keep its validated raw identity above instead of reusing its checksum
        # as though it described the newly grouped records.
        derived.pop("reconciliation", None)
        derived["lsda_grouping"] = derivation
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".lsda-grouping-", dir=output.parent
        ) as directory:
            pending = Path(directory) / output.name
            write_manifest(pending, derived, bound)
            receipt.update(
                derivation=derivation,
                manifest_sha256=sha256(pending),
                full_native_dataset_reproduction=False,
            )
            write_json(Path(directory) / protocol.name, receipt)
            if output.exists() or protocol.exists():
                raise FileExistsError("LSDA destination appeared during preparation")
            os.replace(Path(directory) / protocol.name, protocol)
            os.replace(pending, output)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-map", type=Path)
    args = parser.parse_args()
    category_map = (
        json.loads(args.category_map.read_text()) if args.category_map else None
    )
    receipt = LSDAInventoryBuilder(category_map).build(
        args.input, args.pairs, args.output
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "rows": receipt["output_rows"],
                "groups": len(receipt["groups"]),
                "unused_source_rows": len(receipt["unused_source_ids"]),
            }
        )
    )
