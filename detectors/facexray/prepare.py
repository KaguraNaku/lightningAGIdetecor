"""Compile explicit training bindings for published FFBlend supervision banks."""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from data.manifest import read_manifest, sha256
from .bank import write_bank


def prepare_bank(*, manifest, landmarks_path, neighbors_path, policy, count, output):
    manifest, landmarks_path = Path(manifest).resolve(), Path(landmarks_path).resolve()
    header, records = read_manifest(manifest)
    if header["split"] != "train":
        raise ValueError("FaceXray bank preparation requires a train manifest")
    # Explicit offline import of caller-selected trusted source assets. Runtime
    # and inference never unpickle or retain these legacy dictionaries.
    with landmarks_path.open("rb") as stream:
        native_landmarks = pickle.load(stream)
    keys = [row.metadata.get("facexray_bank_id") for row in records]
    if any(not isinstance(key, str) or not key for key in keys) or len(
        set(keys)
    ) != len(keys):
        raise ValueError(
            "Each training row needs a unique explicit facexray_bank_id metadata binding"
        )
    lookup = {key: index for index, key in enumerate(keys)}
    if any(key not in native_landmarks for key in keys):
        raise ValueError("FaceXray manifest binding has no source landmark record")
    points = [native_landmarks[key] for key in keys]
    origin = {
        "landmarks": {"path": str(landmarks_path), "sha256": sha256(landmarks_path)},
        "background_selection": "Exactly explicit train-manifest rows, in manifest order; not the original all-bank global shuffle",
        "native_landmark_count": len(native_landmarks),
        "selected_landmark_count": len(keys),
        "neighbor_policy": policy,
        "label_transport": "Explicit manifest0/1 replaces source path-component6 inference; mixed background labels retained",
        "coordinate_policy": "Native float32(81,2), including out-of-range coordinates; no clamp or rescale",
        "full_original_protocol_reproduced": False,
    }
    if policy == "provided":
        if neighbors_path is None:
            raise ValueError(
                "Provided FaceXray neighbor policy requires --neighbors-pickle"
            )
        neighbors_path = Path(neighbors_path).resolve()
        with neighbors_path.open("rb") as stream:
            native_neighbors = pickle.load(stream)
        neighbors = {}
        for index, key in enumerate(keys):
            if key in native_neighbors:
                values = native_neighbors[key]
                if not isinstance(values, list) or not values:
                    raise ValueError(
                        "A present native neighbor entry must be a nonempty list"
                    )
                if any(value not in lookup for value in values):
                    raise ValueError(
                        "Native FaceXray foreground outside train inventory: supply complete explicit bindings or deliberately prepare training_kdtree policy; no silent neighbor filtering"
                    )
                neighbors[index] = [lookup[value] for value in values]
        origin["neighbors"] = {
            "path": str(neighbors_path),
            "sha256": sha256(neighbors_path),
        }
        origin["native_neighbor_parent_count"] = len(native_neighbors)
        origin["provided_neighbor_lists_unchanged"] = True
    elif policy == "training_kdtree":
        if neighbors_path is not None:
            raise ValueError(
                "training_kdtree must not silently ignore a provided neighbor pickle"
            )
        from scipy.spatial import KDTree

        parents = [index for index, row in enumerate(records) if row.label == 0]
        if type(count) is not int or not 1 <= count < len(parents):
            raise ValueError(
                "Neighbor count must be positive and smaller than real training inventory"
            )
        values = np.array([points[index].flatten() for index in parents])
        tree = KDTree(values)
        neighbors = {}
        for index, value in enumerate(values):
            _, selected = tree.query(value, k=count + 1)
            # Native helper removes the FIRST result, not every self reference;
            # equidistant cases can retain a self edge and must not be repaired.
            neighbors[parents[index]] = [parents[int(i)] for i in selected[1:]]
        origin.update(
            neighbor_count=count,
            neighbor_pool="Manifest real0 rows in manifest order; all mixed0/1 backgrounds remain",
            protocol_change="EXPLICIT train-scoped KDTree preparation using the native query/order arithmetic; different bank inventory from published full FF++ assets",
            provided_neighbor_lists_unchanged=False,
        )
    else:
        raise ValueError("Unknown FaceXray neighbor preparation policy")
    return write_bank(
        output,
        manifest_path=manifest,
        records=records,
        landmarks=points,
        neighbors=neighbors,
        origin=origin,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--landmarks-pickle", type=Path, required=True)
    parser.add_argument("--neighbors-pickle", type=Path)
    parser.add_argument(
        "--neighbor-policy", choices=["provided", "training_kdtree"], default="provided"
    )
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_bank(
        manifest=args.manifest,
        landmarks_path=args.landmarks_pickle,
        neighbors_path=args.neighbors_pickle,
        policy=args.neighbor_policy,
        count=args.neighbors,
        output=args.output,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
