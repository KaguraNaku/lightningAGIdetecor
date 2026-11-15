"""Native RGB256/NPY quadruples with explicit sources and fatal error receipts."""

from copy import deepcopy
import json
import os
from pathlib import Path

import cv2
import numpy as np

from detectors._shared.randomness import ArrayIdentity, RandomState
from .generation import DOMAINS, ProDetGenerator, ProDetQuadCollator, select_foreground
from .protocol import ProDetSourcePlan, verify_file


class ProDetDataError(RuntimeError):
    marker = "PRODET_DATA_FAILURE:"

    def __init__(self, receipt):
        self.receipt = receipt if isinstance(receipt, dict) else None
        super().__init__(
            self.marker + json.dumps(receipt)
            if self.receipt is not None
            else str(receipt)
        )


def metadata(row):
    return {
        "id": row.id,
        "dataset": row.dataset,
        "split": row.split,
        "video_id": row.video_id,
        "frame_index": row.frame_index,
        "category": row.category,
        "attributes": deepcopy(row.metadata),
    }


def array_identity(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": ArrayIdentity.digest(value),
    }


class ProDetDataset:
    def __init__(self, base, *, neighbors):
        self.plan = ProDetSourcePlan(base, neighbors=neighbors)
        self.root, self.header = self.plan.root, self.plan.header
        self.manifest_path = self.plan.manifest_path
        self.records, self.source_records = self.plan.records, self.plan.source_records
        self.landmark_arrays = {}
        # Validate all real sources, including excluded/unused neighbor members.
        # No loaded arrays or file handles survive construction or worker pickle.
        for row in self.source_records:
            if row.label != 0:
                continue
            try:
                points, descriptor = self._landmarks(row.id)
                self.landmark_arrays[row.id] = descriptor
                del points
            except Exception as error:
                raise ProDetDataError(
                    {
                        "phase": "binding",
                        "source_id": row.id,
                        "operation": "landmark_decode",
                        "attempts": 1,
                        "substitution_performed": False,
                        "binding": self.plan.bindings[row.id],
                        "error": f"{type(error).__name__}: {error}",
                    }
                ) from error
        self._generator = ProDetGenerator()
        self._pid, self._control_stamps = None, {}

    def __len__(self):
        return len(self.plan.pairs)

    def __getstate__(self):
        # Albu transforms are importable/pickleable; keep their configuration.
        return {**self.__dict__, "_pid": None, "_control_stamps": {}}

    def _check_controls(self):
        paths = (self.manifest_path, Path(self.plan.neighbor_identity["path"]))
        stamps = {}
        for path in paths:
            info = path.stat()
            stamps[str(path)] = (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            )
        if self._pid != os.getpid() or stamps != self._control_stamps:
            self.plan.verify_control_files()
            if self._pid != os.getpid():
                cv2.setNumThreads(0)
            self._pid, self._control_stamps = os.getpid(), stamps

    def _image(self, source_id):
        path = verify_file(self.root, self.plan.bindings[source_id]["image"])
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(
                f"OpenCV could not decode requested ProDet source:{source_id}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

    def _landmarks(self, source_id):
        identity = self.plan.bindings[source_id]["landmarks"]
        path = verify_file(self.root, identity)
        points = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(points, np.ndarray)
            or points.shape not in ((68, 2), (81, 2))
            or points.dtype.kind not in "iuf"
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                "ProDet requires finite numeric 68/81x2 native landmark arrays"
            )
        descriptor = {
            **identity,
            "shape": list(points.shape),
            "dtype": str(points.dtype),
            "coordinate_size": 256,
            "rescaled": False,
        }
        # Exact native filesystem operation: truncate into int32, no rescaling.
        return points.astype(np.int32), descriptor

    def selection_receipt(self):
        return {
            **self.plan.selection_receipt(),
            "landmark_arrays": deepcopy(self.landmark_arrays),
            "dataset_length_unit": "quadruple;records enumerate four distinct view identities per quadruple",
            "pixel_decoder": "Native cv2.imread/BGR-to-RGB/cubic256;filesystem image sources only",
            "landmark_decoder": "Native unscaled filesystem NPY then int32;missing,nonfinite,invalid shape/type are fatal instead of zero fill",
            "failure_policy": "One requested-source attempt;no recursive or neighboring replacement;receipt preserves pair/source/bindings/RNG",
            "worker_policy": "Spawn-pickle dataset/transforms;shared loader seeds Python/NumPy from Torch worker seed. Conditional recipe parity,not original upstream or8-worker global RNG reproduction.",
        }

    def __getitem__(self, index):
        pair = self.plan.pairs[index]
        # Normalize Python's negative indexing for the static view inventory.
        if index < 0:
            index += len(self)
        before, selected, operation = RandomState.capture(), None, "control_file_verification"
        try:
            self._check_controls()
            real_id, fake_id = pair["real_id"], pair["fake_id"]
            operation = "landmark_decode"
            points, point_identity = self._landmarks(real_id)
            operation = "deepfake_image_decode"
            deepfake = self._image(fake_id)
            operation = "real_image_decode"
            real = self._image(real_id)
            landmark_key = self.plan.bindings[real_id]["native_landmark_path"]

            def load_foreground():
                nonlocal selected, operation
                operation = "foreground_selection"
                key = select_foreground(landmark_key, self.plan.neighbors)
                selected = self.plan.by_landmark[key]
                operation = "foreground_image_decode"
                foreground = self._image(selected)
                # Native generate_bi reloads the background's landmarks here.
                operation = "bi_background_landmark_decode"
                reloaded, _ = self._landmarks(real_id)
                if not np.array_equal(points, reloaded):
                    raise ValueError(
                        "ProDet background landmarks changed between SBI and BI"
                    )
                operation = "bi_synthesis_and_shared_augmentation"
                return foreground

            operation = "sbi_synthesis"
            result = self._generator.generate(
                real, deepfake, points, load_foreground=load_foreground
            )
            views = self.records[index * 4 : (index + 1) * 4]
            result["metadata"] = {
                name: metadata(row)
                for (name, _), row in zip(DOMAINS, views, strict=True)
            }
            result["generation"] = {
                "quadruple_id": pair["id"],
                "pair": deepcopy(pair),
                "real_source": metadata(self.plan.by_id[real_id]),
                "deepfake_source": metadata(self.plan.by_id[fake_id]),
                "foreground_source": metadata(self.plan.by_id[selected]),
                "foreground_self_fallback": landmark_key not in self.plan.neighbors,
                "input_bindings": {
                    source_id: self.plan.bindings[source_id]
                    for source_id in (real_id, fake_id, selected)
                },
                "landmarks": point_identity,
                "domain_labels": dict(DOMAINS),
                "masks": {
                    name: array_identity(value)
                    for name, value in result["masks"].items()
                },
                "normalized_views": {
                    name: array_identity(result[name][0].numpy()) for name, _ in DOMAINS
                },
                "rng_before": before,
                "rng_after": RandomState.capture(),
                "substitution_performed": False,
                "source_short_list_clamping": pair["real_clamped"]
                or pair["fake_clamped"],
            }
            return result
        except Exception as error:
            raise ProDetDataError(
                {
                    "phase": "getitem",
                    "requested_quadruple": deepcopy(pair),
                    "dataset": self.header["dataset"],
                    "split": self.header["split"],
                    "foreground_id": selected,
                    "operation": operation,
                    "attempts": 1,
                    "substitution_performed": False,
                    "real_binding": self.plan.bindings[pair["real_id"]],
                    "fake_binding": self.plan.bindings[pair["fake_id"]],
                    "rng_before": before,
                    "rng_after": RandomState.capture(),
                    "error": f"{type(error).__name__}: {error}",
                }
            ) from error


class ProDetDataCollator(ProDetQuadCollator):
    """Keep source-level generation evidence alongside all four ordered views."""

    def __call__(self, samples):
        batch = super().__call__(samples)
        receipts, seen = [], set()
        for sample in samples:
            receipt = sample["generation"]
            quad_id = receipt["quadruple_id"]
            if quad_id in seen or any(
                sample["metadata"][name]["id"] != f"{quad_id}:{name}"
                for name, _ in DOMAINS
            ):
                raise ValueError(
                    "ProDet generation/view identities are repeated or misaligned"
                )
            seen.add(quad_id)
            receipts.append(deepcopy(receipt))
        return {**batch, "generation_receipts": receipts}
