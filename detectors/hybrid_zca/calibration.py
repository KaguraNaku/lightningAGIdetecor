"""Source-order GAP fitting and a disk-backed native KMeans patch matrix."""

import os
from pathlib import Path
import shutil

import numpy as np
import torch

from data.manifest import sha256
from engine.runtime import atomic_save
from .statistics import DIMENSION, PATCHES


class HybridZCACalibrator:
    def __init__(
        self,
        statistics,
        output,
        *,
        feature_budget_mib=256,
        patch_budget_gib=1,
        disk_reserve_gib=8,
    ):
        for value in (feature_budget_mib, patch_budget_gib, disk_reserve_gib):
            if type(value) is not int or value < 1:
                raise ValueError("Hybrid ZCA storage budgets must be positive integers")
        self.statistics = statistics
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.path = self.output / "real-primary-patches.f32"
        self.stream = self.path.open("xb")
        self.feature_budget = feature_budget_mib * 1024**2
        self.patch_budget = patch_budget_gib * 1024**3
        self.disk_reserve = disk_reserve_gib * 1024**3
        self.features = {layer: [] for layer in statistics.layers}
        self.labels = []
        self.real_count = self.fake_count = self.feature_bytes = self.patch_bytes = 0
        self.finished = False

    def add(self, real, fake, real_patches):
        if self.finished or self.stream.closed:
            raise ValueError("Hybrid ZCA calibration collection has already closed")
        nreal = self.statistics.validate_features(real)
        nfake = self.statistics.validate_features(fake)
        if not nreal + nfake:
            raise ValueError("Hybrid ZCA cannot collect an empty calibration batch")
        if (
            not isinstance(real_patches, torch.Tensor)
            or real_patches.shape != (nreal, PATCHES, DIMENSION)
            or real_patches.dtype != torch.float32
            or not torch.isfinite(real_patches).all()
        ):
            raise ValueError(
                "Hybrid ZCA requires finite N x256 x1024 real patch tokens"
            )
        extra_features = (nreal + nfake) * len(self.features) * DIMENSION * 4
        extra_patches = nreal * PATCHES * DIMENSION * 4
        if self.feature_bytes + extra_features > self.feature_budget:
            raise OSError("Hybrid ZCA GAP buffer budget would be exceeded")
        if self.patch_bytes + extra_patches > self.patch_budget:
            raise OSError("Hybrid ZCA preserved patch budget would be exceeded")
        if shutil.disk_usage(self.output).free - extra_patches < self.disk_reserve:
            raise OSError("Hybrid ZCA patch write would cross the disk reserve")
        for layer in self.statistics.layers:
            # Match source append order: real batch, fake batch, repeated.
            self.features[layer].append(real[layer].clone().detach().cpu())
            self.features[layer].append(fake[layer].clone().detach().cpu())
        self.labels.extend([torch.zeros(nreal), torch.ones(nfake)])
        if nreal:
            values = np.ascontiguousarray(real_patches.detach().cpu().numpy())
            written = self.stream.write(memoryview(values).cast("B"))
            assert written == extra_patches
        self.stream.flush()
        self.real_count += nreal
        self.fake_count += nfake
        self.feature_bytes += extra_features
        self.patch_bytes += extra_patches

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
        atomic_save(
            self.output / "collection.json",
            {
                "real_images": self.real_count,
                "fake_images": self.fake_count,
                "GAP_bytes": self.feature_bytes,
                "primary_real_patch_bytes": self.patch_bytes,
                "primary_real_patch_path": str(self.path),
                "dtype": "float32",
                "columns": DIMENSION,
                "patches_per_image": PATCHES,
                "feature_budget_bytes": self.feature_budget,
                "patch_budget_bytes": self.patch_budget,
                "disk_reserve_bytes": self.disk_reserve,
                "finished": self.finished,
                "raw_inputs_preserved": True,
            },
        )

    @torch.no_grad()
    def finish(self):
        if self.finished:
            raise ValueError("Hybrid ZCA calibration has already finished")
        self.close()
        if self.real_count < 2 or self.fake_count < 1:
            raise ValueError(
                "Hybrid ZCA calibration needs at least2 real and1 fake images"
            )
        count = self.real_count * PATCHES
        if count < self.statistics.memory_size:
            raise ValueError(
                "Hybrid ZCA needs enough real patches for its native memory bank"
            )
        assert self.path.stat().st_size == self.patch_bytes == count * DIMENSION * 4
        device = getattr(
            self.statistics, f"real_mean_{self.statistics.layers[0]}"
        ).device
        labels = torch.cat(self.labels, dim=0).to(device)
        matrices = {
            layer: torch.cat(rows, dim=0).to(device)
            for layer, rows in self.features.items()
        }
        atomic_save(
            self.output / "GAP-inputs.pth",
            {"features": self.features, "labels": self.labels},
            torch_format=True,
        )
        eigenvalues = self.statistics.fit_zca(matrices, labels)
        # The inactive source micro module still builds this bank. Preserve its
        # state and fallback RNG behavior; only the matrix storage changes.
        from sklearn.cluster import MiniBatchKMeans
        import sklearn

        before = sha256(self.path)
        patches = np.memmap(
            self.path, dtype=np.float32, mode="r+", shape=(count, DIMENSION)
        )
        fallback = None
        try:
            estimator = MiniBatchKMeans(
                n_clusters=self.statistics.memory_size,
                batch_size=1024,
                max_iter=100,
                random_state=42,
                verbose=1,
            )
            estimator.fit(patches)
            centers = torch.from_numpy(estimator.cluster_centers_).float()
        except Exception as error:
            fallback = f"{type(error).__name__}: {error}"
            indices = torch.randperm(count)[: self.statistics.memory_size]
            centers = torch.from_numpy(np.asarray(patches[indices.numpy()])).float()
        getattr(self.statistics, f"memory_bank_{self.statistics.layers[0]}").copy_(
            centers
        )
        patches.flush()
        after = sha256(self.path)
        if before != after:
            raise ValueError("KMeans modified the preserved source patch matrix")
        self.finished = True
        self.close()
        result = {
            "real_images": self.real_count,
            "fake_images": self.fake_count,
            "selected_layers": list(self.statistics.layers),
            "ridge": 0.01,
            "full_dimension": DIMENSION,
            "eigenvalues": eigenvalues,
            "memory_centers": self.statistics.memory_size,
            "memory_primary_layer": self.statistics.layers[0],
            "memory_input_shape": [count, DIMENSION],
            "memory_input_sha256": before,
            "memory_input_unchanged": True,
            "KMeans_random_state": 42,
            "KMeans_batch_size": 1024,
            "KMeans_max_iter": 100,
            "KMeans_fallback": fallback,
            "sklearn": sklearn.__version__,
            "active_score": "macro-only ZCA dot product; micro always zero",
            "dead_fake_and_nonprimary_patch_buffers_retained": False,
            "source_real_patch_order_preserved": True,
            "raw_calibration_inputs_preserved": True,
            "full_experiment_reproduction": False,
        }
        atomic_save(self.output / "calibration.json", result)
        self.features.clear()
        self.labels.clear()
        return result
