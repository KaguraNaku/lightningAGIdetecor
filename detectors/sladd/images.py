"""Unpaired source validation/evaluation with native filesystem RGB decoding."""

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data.manifest import sha256
from .dataset import metadata, SLADDDataError
from .protocol import identity


class SLADDImageDataset:
    def __init__(self, base, *, stage):
        if (
            stage not in {"validation", "evaluation"}
            or (stage == "validation" and base.header["split"] != "val")
            or (stage == "evaluation" and base.header["split"] not in {"val", "test"})
        ):
            raise ValueError(
                "SLADD unpaired images require matching validation/evaluation role"
            )
        self.base, self.stage = base, stage
        self.root, self.header = Path(base.root).resolve(), deepcopy(base.header)
        self.manifest_path, self.manifest_sha256 = (
            base.manifest_path,
            sha256(base.manifest_path),
        )
        self.records = deepcopy(base.records)
        self.bindings = {row.id: identity(self.root, row.path) for row in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        try:
            if sha256(self.manifest_path) != self.manifest_sha256:
                raise ValueError("SLADD unpaired manifest changed")
            if identity(self.root, row.path) != self.bindings[row.id]:
                raise ValueError("SLADD unpaired source changed")
            if row.storage == "image":
                image = cv2.imread(str(self.root / row.path))
                if image is None:
                    raise ValueError("SLADD could not decode requested image")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = np.asarray(self.base._parquet_image(row).convert("RGB"))
            return {
                "image": Image.fromarray(image),
                "label": row.label,
                "metadata": metadata(row),
            }
        except Exception as error:
            raise SLADDDataError(
                {
                    "phase": "unpaired_image",
                    "stage": self.stage,
                    "requested_source": metadata(row),
                    "attempts": 1,
                    "substitution_performed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            ) from error

    def selection_receipt(self):
        return {
            "stage": self.stage,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "sources": len(self),
            "bindings": self.bindings,
            "decoder": "Filesystem OpenCV BGR->RGB;Parquet PIL bytes as native tuple branch;collation cubic256/half normalization,no augmentation",
            "failure": "Fatal requested-source error, no recursion/filter/replacement",
            "upstream_discovery_or8_worker_RNG_reproduced": False,
        }
