"""Method-owned masks and native targets, composed over immutable shared images."""

from dataclasses import dataclass
import cv2
import numpy as np
import torch
from torch.utils.data import Sampler
from data.loader import ImageCollator
from data.manifest import relative_path, sha256
from detectors._shared.transforms import ImageTransform


# get_boundary and split_images_by_patch below retain the author's float32 OpenCV
# operations, including bitwise_not(float32); no binary-count shortcut is applied.
def get_boundary(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    boundary = mask / 255.0
    boundary = 4 * boundary * (1.0 - boundary)
    return boundary


def split_images_by_patch(mask_image, patch_size, mode="normal", need_boundary=False):
    if mode == "resize":
        mask_image = cv2.resize(mask_image, (224, 224))

    _, b_image = cv2.threshold(mask_image, 40, 255, cv2.THRESH_BINARY)
    height, width = b_image.shape

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    labels = []
    if_boundaries = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_y = i * patch_size
            start_x = j * patch_size
            patch = b_image[
                start_y : start_y + patch_size, start_x : start_x + patch_size
            ]
            white_pixels = (
                cv2.findNonZero(patch).shape[0]
                if cv2.findNonZero(patch) is not None
                else 0
            )
            black_pixels = (
                cv2.findNonZero(cv2.bitwise_not(patch)).shape[0]
                if cv2.findNonZero(cv2.bitwise_not(patch)) is not None
                else 0
            )
            total_pixels = black_pixels + white_pixels
            label = 1 if (white_pixels / total_pixels > 0.1) else 0  # 0 real  1 fake

            labels.append(label)
            if need_boundary:
                if_boundary = (
                    1
                    if (
                        white_pixels / total_pixels > 0.1
                        and white_pixels != total_pixels
                    )
                    else 0
                )  # 0 real  1 fake
                if_boundaries.append(if_boundary)
    if need_boundary:
        return labels, if_boundaries
    else:
        return labels


class ForAdaDataset:
    def __init__(self, base, *, require_localized=False):
        self.base = base
        self.records, self.header = base.records, base.header
        self.root, self.manifest_path = base.root, base.manifest_path
        self.bindings = {}
        localized = 0
        for row in self.records:
            if row.label == 0:
                continue
            entry = row.metadata.get("forada_mask")
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError(
                    "Fake ForAda training images require metadata.forada_mask {path, sha256}"
                )
            path = (self.root / relative_path(entry["path"])).resolve()
            if (
                not path.is_relative_to(self.root.resolve())
                or sha256(path) != entry["sha256"]
            ):
                raise ValueError("ForAda mask path escapes root or checksum differs")
            self.bindings[row.id] = dict(entry)
            if require_localized:
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError("Cannot decode author-supervision mask")
                localized += int(mask.min() != mask.max())
        if require_localized and (
            self.header["dataset"] != "FaceForensics++" or localized == 0
        ):
            raise ValueError(
                "ForAda author FF++ training needs localized manipulation masks; whole-image labels are not boundary supervision"
            )
        self.supervision_summary = {
            "fake_masks": len(self.bindings),
            "localized_masks": localized if require_localized else None,
            "policy": "localized_author_masks"
            if require_localized
            else "explicit_bound_masks",
        }

    def __len__(self):
        return len(self.base)

    def selection_receipt(self):
        return {
            "mask_policy": "real_zero;fake_explicit_sha256_bound;no_synthesis_or_substitution",
            "supervision": self.supervision_summary,
            "mask_bindings": self.bindings,
        }

    def __getitem__(self, index):
        sample = self.base[index]
        if sample is None:
            return None
        row = self.records[index]
        if row.label == 0:
            mask = np.zeros((256, 256), dtype=np.float32)
        else:
            binding = self.bindings[row.id]
            path = self.root / relative_path(binding["path"])
            if sha256(path) != binding["sha256"]:
                raise ValueError("ForAda mask changed after binding")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Cannot decode bound ForAda mask")
            mask = cv2.resize(mask, (256, 256)).astype(np.float32)
        return {**sample, "forada_mask": mask}


@dataclass
class ForAdaCollator:
    preprocessing: dict
    training: bool = False

    def __call__(self, samples):
        valid = [s for s in samples if s is not None]
        batch = ImageCollator(transform=ImageTransform(self.preprocessing, False))(
            valid
        )
        if batch is None or not self.training:
            return batch
        if any("forada_mask" not in s for s in valid):
            raise ValueError("ForAda training requires the method dataset adapter")
        targets = {
            "xray": [],
            "patch_label": [],
            "clip_patch_label": [],
            "if_boundary": [],
        }
        for sample in valid:
            mask = sample["forada_mask"]
            patch, boundary = split_images_by_patch(mask.copy(), 16, need_boundary=True)
            targets["xray"].append(
                torch.from_numpy(get_boundary(mask.copy())).unsqueeze(0)
            )
            targets["patch_label"].append(torch.tensor(patch, dtype=torch.long))
            targets["clip_patch_label"].append(
                torch.tensor(
                    split_images_by_patch(mask.copy(), 14, mode="resize"),
                    dtype=torch.long,
                )
            )
            targets["if_boundary"].append(torch.tensor(boundary, dtype=torch.long))
        return {
            **batch,
            **{key: torch.stack(values) for key, values in targets.items()},
        }


class WholeImageForAdaDataset:
    """Fully generated images have full-image masks; genuine images have zero masks.

    This explicit UFD extension is not a claim of annotated manipulation boundaries.
    The native collator/loss, including its boundary mathematics, remains unchanged.
    """

    def __init__(self, base):
        if (
            base.header["dataset"] != "UniversalFakeDetect"
            or base.header["split"] != "train"
        ):
            raise ValueError(
                "Whole-image supervision is restricted to declared UFD training"
            )
        self.base, self.records, self.header = base, base.records, base.header
        self.root, self.manifest_path = base.root, base.manifest_path
        if any(
            r.label == 1 and r.metadata.get("generator") != "ProGAN"
            for r in self.records
        ):
            raise ValueError("Fake UFD rows require explicit generator=ProGAN metadata")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        if sample is None:
            return None
        mask = np.full(
            (256, 256), 255.0 if self.records[index].label else 0.0, dtype=np.float32
        )
        return {**sample, "forada_mask": mask}

    def selection_receipt(self):
        return {
            "mask_policy": "UFD explicit whole-image synthetic255/genuine0;derived from image labels",
            "boundary_claim": "No localized manipulation boundary annotation exists;native boundary targets may degenerate",
            "source_manifest": str(self.manifest_path),
        }


class FakeGuardSampler(Sampler):
    """Shuffle without replacement and reserve at least one fake per native batch."""

    def __init__(self, records, batch_size, seed):
        self.real = [i for i, r in enumerate(records) if r.label == 0]
        self.fake = [i for i, r in enumerate(records) if r.label == 1]
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.size = len(records)
        if len(self.fake) < (self.size + batch_size - 1) // batch_size:
            raise ValueError("Not enough fake images to guard every ForAda batch")

    def __len__(self):
        return self.size

    def state_dict(self):
        return {
            "version": 1,
            "seed": self.seed,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state):
        if (
            state.get("version") != 1
            or state.get("seed") != self.seed
            or state.get("batch_size") != self.batch_size
            or type(state.get("epoch")) is not int
            or state["epoch"] < 0
        ):
            raise ValueError("ForAda sampler continuation protocol differs")
        self.epoch = state["epoch"]

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        real = [
            self.real[i] for i in torch.randperm(len(self.real), generator=rng).tolist()
        ]
        fake = [
            self.fake[i] for i in torch.randperm(len(self.fake), generator=rng).tolist()
        ]
        ri = fi = 0
        while ri + fi < self.size:
            left = self.size - ri - fi
            size = min(self.batch_size, left)
            steps = (left + self.batch_size - 1) // self.batch_size
            nf, nr = len(fake) - fi, len(real) - ri
            take_fake = max(1, round(size * nf / left), size - nr)
            take_fake = min(take_fake, size, nf - (steps - 1))
            take_real = size - take_fake
            if not (0 <= take_real <= nr and 1 <= take_fake <= nf):
                raise ValueError(
                    "Cannot construct native ForAda batch without replacement"
                )
            batch = real[ri : ri + take_real] + fake[fi : fi + take_fake]
            for i in torch.randperm(len(batch), generator=rng).tolist():
                yield batch[i]
            ri += take_real
            fi += take_fake
