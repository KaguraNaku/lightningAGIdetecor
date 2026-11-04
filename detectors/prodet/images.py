"""Generic native-entrypoint RGB images with method-owned filesystem decoding."""

from copy import deepcopy
import os

import cv2
import numpy as np
from PIL import Image

from data.manifest import sha256
from .dataset import ProDetDataError, metadata
from .protocol import file_identity, verify_file


class ProDetImageDataset:
    def __init__(self, base, *, stage):
        if (
            stage not in {"train", "validation", "evaluation"}
            or (stage == "train" and base.header["split"] != "train")
            or (stage == "validation" and base.header["split"] != "val")
            or (stage == "evaluation" and base.header["split"] not in {"val", "test"})
        ):
            raise ValueError(
                "ProDet generic image adapter requires explicit matching data roles"
            )
        self.base, self.stage = base, stage
        self.root, self.header = base.root, deepcopy(base.header)
        self.manifest_path = base.manifest_path
        self.manifest_sha256 = sha256(self.manifest_path)
        self.records = deepcopy(base.records)
        self.source_records = self.records
        self.bindings = {
            row.id: file_identity(self.root, row.path)
            for row in self.records
            if row.storage == "image"
        }
        self._pid = None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {**self.__dict__, "_pid": None}

    def __getitem__(self, index):
        row = self.records[index]
        try:
            if self._pid != os.getpid():
                if sha256(self.manifest_path) != self.manifest_sha256:
                    raise ValueError("ProDet generic parent manifest changed")
                cv2.setNumThreads(0)
                self._pid = os.getpid()
            if row.storage == "image":
                path = verify_file(self.root, self.bindings[row.id])
                pixels = cv2.imread(str(path))
                if pixels is None:
                    raise ValueError("OpenCV could not decode requested ProDet image")
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            else:
                # The native parquet branch decodes PIL bytes,then BGR->RGB.
                pixels = np.asarray(self.base._parquet_image(row).convert("RGB"))
            return {
                "image": Image.fromarray(pixels),
                "label": row.label,
                "metadata": metadata(row),
            }
        except Exception as error:
            raise ProDetDataError(
                {
                    "phase": "generic_image",
                    "stage": self.stage,
                    "requested_source": metadata(row),
                    "attempts": 1,
                    "substitution_performed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            ) from error

    def selection_receipt(self):
        return {
            "data_profile": "generic_native_images",
            "stage": self.stage,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "source_count": len(self),
            "filesystem_bindings": self.bindings,
            "decode": "Native OpenCV BGR-to-RGB for files;PIL RGB for parquet bytes;resize/augmentation/half-normalization at collation",
            "failure": "Fatal requested-source failure instead of native recursion/zero-image replacement",
            "upstream_discovery_or_process_RNG_reproduced": False,
        }
