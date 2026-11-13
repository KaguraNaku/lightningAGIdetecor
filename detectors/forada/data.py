"""Method-owned masks and native targets, composed over immutable shared images."""
from dataclasses import dataclass
import cv2
import numpy as np
import torch
from data.loader import ImageCollator
from data.manifest import relative_path, sha256
from detectors._shared.transforms import ImageTransform

# get_boundary and split_images_by_patch below retain the author's float32 OpenCV
# operations, including bitwise_not(float32); no binary-count shortcut is applied.
def get_boundary(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    boundary = mask / 255.
    boundary = 4 * boundary * (1. - boundary)
    return boundary

def split_images_by_patch(mask_image, patch_size, mode='normal', need_boundary=False):
    if mode == 'resize':
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
            patch = b_image[start_y:start_y + patch_size, start_x:start_x + patch_size]
            white_pixels = cv2.findNonZero(patch).shape[0] if cv2.findNonZero(
                patch) is not None else 0
            black_pixels = cv2.findNonZero(cv2.bitwise_not(patch)).shape[0] if cv2.findNonZero(
                cv2.bitwise_not(patch)) is not None else 0
            total_pixels = black_pixels + white_pixels
            label = 1 if (white_pixels / total_pixels > 0.1 ) else 0  # 0 real  1 fake

            labels.append(label)
            if need_boundary:
                if_boundary = 1 if (white_pixels / total_pixels > 0.1 and white_pixels != total_pixels) else 0  # 0 real  1 fake
                if_boundaries.append(if_boundary)
    if need_boundary:
        return labels, if_boundaries
    else:
        return labels

class ForAdaDataset:
    def __init__(self, base):
        self.base = base
        self.records, self.header = base.records, base.header
        self.root, self.manifest_path = base.root, base.manifest_path
        self.bindings = {}
        for row in self.records:
            if row.label == 0:
                continue
            entry = row.metadata.get("forada_mask")
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("Fake ForAda training images require metadata.forada_mask {path, sha256}")
            path = (self.root / relative_path(entry["path"])).resolve()
            if not path.is_relative_to(self.root.resolve()) or sha256(path) != entry["sha256"]:
                raise ValueError("ForAda mask path escapes root or checksum differs")
            self.bindings[row.id] = dict(entry)

    def __len__(self):
        return len(self.base)

    def selection_receipt(self):
        return {"mask_policy": "real_zero;fake_explicit_sha256_bound;no_synthesis_or_substitution",
                "mask_bindings": self.bindings}

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
        batch = ImageCollator(transform=ImageTransform(self.preprocessing, False))(valid)
        if batch is None or not self.training:
            return batch
        if any("forada_mask" not in s for s in valid):
            raise ValueError("ForAda training requires the method dataset adapter")
        targets = {"xray": [], "patch_label": [], "clip_patch_label": [], "if_boundary": []}
        for sample in valid:
            mask = sample["forada_mask"]
            patch, boundary = split_images_by_patch(mask.copy(), 16, need_boundary=True)
            targets["xray"].append(torch.from_numpy(get_boundary(mask.copy())).unsqueeze(0))
            targets["patch_label"].append(torch.tensor(patch, dtype=torch.long))
            targets["clip_patch_label"].append(torch.tensor(split_images_by_patch(mask.copy(), 14, mode="resize"), dtype=torch.long))
            targets["if_boundary"].append(torch.tensor(boundary, dtype=torch.long))
        return {**batch, **{key: torch.stack(values) for key, values in targets.items()}}
