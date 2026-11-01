"""Decode images, tolerating isolated failures with a consecutive-error limit."""

from collections import OrderedDict
from io import BytesIO
import os
import logging
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from .config import DataConfig
from .manifest import read_manifest, read_metadata, relative_path, sha256

logger = logging.getLogger(__name__)


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path | str,
        config: DataConfig,
        *,
        verify_hashes: bool = True,
        decode_error_patience: int = 10,
    ):
        if (
            isinstance(decode_error_patience, bool)
            or not isinstance(decode_error_patience, int)
            or decode_error_patience < 1
        ):
            raise ValueError("decode_error_patience must be a positive integer")
        self.decode_error_patience = decode_error_patience
        self.consecutive_decode_errors = 0
        self.manifest_path = Path(manifest).resolve()
        self.header, self.records = read_manifest(manifest)
        read_metadata(manifest, self.header)
        self.settings = config.dataset(self.header["dataset"])
        self.root = self.settings["root"]
        # Decoding settings are frozen in the manifest, not taken from a newer config.
        self.image_column = self.header["settings"].get("image_column", "image")
        self._pid = os.getpid()
        self._groups = OrderedDict()
        self._cache_capacity = 1
        for source, identity in self.header["sources"].items():
            path = self.root / relative_path(source)
            if path.stat().st_size != identity["size"] or (
                verify_hashes and sha256(path) != identity["sha256"]
            ):
                raise ValueError(f"Data source changed since manifest creation: {path}")
        for record in self.records:
            if (
                record.storage == "parquet"
                and record.path not in self.header["sources"]
            ):
                raise ValueError(f"Unfingerprinted Parquet shard: {record.path}")

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        return {
            **self.__dict__,
            "_groups": OrderedDict(),
            "consecutive_decode_errors": 0,
        }

    def _parquet_image(self, record):
        if self._pid != os.getpid():
            self._groups.clear()
            self._pid = os.getpid()
        key = (record.path, record.row_group)
        if key not in self._groups:
            with pq.ParquetFile(self.root / record.path) as shard:
                table = shard.read_row_group(
                    record.row_group, columns=[self.image_column]
                )
            self._groups[key] = table
            if len(self._groups) > self._cache_capacity:
                self._groups.popitem(last=False)
        self._groups.move_to_end(key)
        cell = self._groups[key][self.image_column][record.row_index].as_py()
        if isinstance(cell, dict):
            if cell.get("bytes") is not None:
                with Image.open(BytesIO(cell["bytes"])) as image:
                    return image.convert("RGB")
            cell = cell.get("path")
        if isinstance(cell, bytes):
            with Image.open(BytesIO(cell)) as image:
                return image.convert("RGB")
        # External image references must also be relocatable, relative to root.
        with Image.open(self.root / relative_path(cell)) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        if self._pid != os.getpid():
            self._groups.clear()
            self._pid = os.getpid()
            self.consecutive_decode_errors = 0
        record = self.records[index]
        try:
            if record.storage == "image":
                with Image.open(self.root / record.path) as source:
                    image = source.convert("RGB")
            else:
                image = self._parquet_image(record)
        except Exception as error:
            self.consecutive_decode_errors += 1
            logger.warning(
                "Cannot decode sample %s (path=%s, row_group=%s, row_index=%s; "
                "consecutive errors %d/%d): %s",
                record.id,
                record.path,
                record.row_group,
                record.row_index,
                self.consecutive_decode_errors,
                self.decode_error_patience,
                error,
            )
            if self.consecutive_decode_errors >= self.decode_error_patience:
                raise RuntimeError(
                    f"Cannot decode sample {record.id}: reached "
                    f"{self.consecutive_decode_errors} consecutive image decode errors "
                    f"(decode_error_patience={self.decode_error_patience}): {error}"
                ) from error
            return None
        self.consecutive_decode_errors = 0
        return {
            "image": image,
            "label": record.label,
            "metadata": {
                "id": record.id,
                "dataset": record.dataset,
                "split": record.split,
                "video_id": record.video_id,
                "frame_index": record.frame_index,
                "category": record.category,
                "attributes": record.metadata,
            },
        }
