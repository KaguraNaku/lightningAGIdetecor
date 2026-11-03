"""Native FF++ filename pairing, fatal decoding, and real-then-fake collation."""

from dataclasses import dataclass
from pathlib import PurePosixPath

import torch
from torch.utils.data import Dataset, Sampler

from detectors._shared.transforms import ImageTransform


def native_key(record):
    path = (
        record.path
        if record.storage == "image"
        else record.metadata.get("native_frame_path")
    )
    if not isinstance(path, str):
        raise ValueError(
            "Hybrid ZCA Parquet rows require explicit native_frame_path metadata"
        )
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if len(parts) < 2:
        raise ValueError(
            "Hybrid ZCA pairing requires video-parent/frame-filename paths"
        )
    return parts[-2], parts[-1]


class HybridPairDataset(Dataset):
    def __init__(self, source):
        self.source = source
        self.header, self.manifest_path = source.header, source.manifest_path
        self.pairs = []
        real_index = {}
        self.overwritten_real_ids = []
        self.unmatched_fake_ids = []
        for index, record in enumerate(source.records):
            if record.label == 0:
                key = native_key(record)
                if key in real_index:
                    self.overwritten_real_ids.append(source.records[real_index[key]].id)
                real_index[key] = index  # Preserve the source's last-real-row wins.
        for index, record in enumerate(source.records):
            if record.label:
                video, frame = native_key(record)
                real = real_index.get((video.split("_")[0], frame))
                if real is None:
                    self.unmatched_fake_ids.append(record.id)
                else:
                    self.pairs.append((real, index))
        if not self.pairs:
            raise ValueError(
                "Hybrid ZCA manifest has no native-matched real/fake pairs"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        real_index, fake_index = self.pairs[index]
        real, fake = self.source[real_index], self.source[fake_index]
        if real is None or fake is None:
            raise RuntimeError(
                f"Hybrid ZCA pair{index} has an undecodable source; no substitution"
            )
        return {"real": real, "fake": fake, "pair_index": index}

    def inventory(self):
        used = {i for pair in self.pairs for i in pair}
        return {
            "available_source_rows": len(self.source.records),
            "matched_pairs": len(self.pairs),
            "matched_source_ids": [self.source.records[i].id for i in sorted(used)],
            "unmatched_source_ids": [
                r.id for i, r in enumerate(self.source.records) if i not in used
            ],
            "unmatched_fake_ids": self.unmatched_fake_ids,
            "overwritten_real_ids": self.overwritten_real_ids,
            "pairs": [
                {
                    "pair_index": i,
                    "real_id": self.source.records[a].id,
                    "fake_id": self.source.records[b].id,
                }
                for i, (a, b) in enumerate(self.pairs)
            ],
            "pairing_rule": "native parent(first underscore segment)/exact frame filename; last real key wins; real reuse retained",
        }


class HybridPairSampler(Sampler):
    """A seed-explicit pair permutation, truncated to native complete batches."""

    def __init__(self, dataset, *, batch_size, seed, shuffle=True):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Hybrid ZCA pair batch size must be positive")
        self.count = len(dataset)
        self.size = self.count // batch_size * batch_size
        if not self.size:
            raise ValueError("Hybrid ZCA has no complete native pair batch")
        self.seed, self.shuffle = seed, shuffle
        self.indices = None

    def __len__(self):
        return self.size

    def __iter__(self):
        self.indices = (
            torch.randperm(
                self.count, generator=torch.Generator().manual_seed(self.seed)
            ).tolist()
            if self.shuffle
            else list(range(self.count))
        )
        return iter(self.indices[: self.size])


@dataclass
class HybridPairCollator:
    preprocessing: dict

    def __call__(self, samples):
        if not samples or any(sample is None for sample in samples):
            raise ValueError("Hybrid ZCA requires complete, decodable source pairs")
        transform = ImageTransform(self.preprocessing, training=False)
        ordered = [sample[side] for side in ("real", "fake") for sample in samples]
        return {
            "pixel_values": torch.stack(
                [transform(sample["image"]) for sample in ordered]
            ),
            "labels": torch.tensor([sample["label"] for sample in ordered]),
            "metadata": [sample["metadata"] for sample in ordered],
            "pair_indices": [sample["pair_index"] for sample in samples],
        }
