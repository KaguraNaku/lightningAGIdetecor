"""Lossless disk-backed patch banks and unchanged LedoitWolf fitting."""

import os
from pathlib import Path
import shutil
import tempfile
import weakref

import torch
from sklearn.covariance import LedoitWolf


class FeatureBank:
    def __init__(self, directory=None):
        directory = directory or ("/var/tmp" if Path("/var/tmp").is_dir() else None)
        self.root = Path(
            tempfile.mkdtemp(prefix=f"remake-uia-bank-{os.getuid()}-", dir=directory)
        )
        self._cleanup = weakref.finalize(self, shutil.rmtree, self.root, True)
        self.chunks = []

    def close(self):
        self._cleanup()
        self.chunks = []

    def append(self, values):
        values = values.detach().cpu().contiguous()
        if values.ndim != 2 or values.shape[1] != 768 or values.dtype != torch.float32:
            raise ValueError("UIA patch banks require float32 (N, 768) features")
        path = self.root / f"{len(self.chunks):08d}.bin"
        values.numpy().tofile(path)
        self.chunks.append((path, values.shape[0]))

    def tensors(self):
        return [
            torch.from_file(
                str(path), shared=False, size=rows * 768, dtype=torch.float32
            ).view(rows, 768)
            if rows
            else torch.empty(0, 768)
            for path, rows in self.chunks
        ]

    def concatenate(self):
        rows = sum(count for _, count in self.chunks)
        if rows == 0:
            raise ValueError("UIA covariance requires a nonempty bank for each class")
        path = self.root / "concatenated.bin"
        with path.open("wb") as destination:
            for source, _ in self.chunks:
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, destination, length=1024 * 1024)
        return torch.from_file(
            str(path), shared=False, size=rows * 768, dtype=torch.float32
        ).view(rows, 768)

    def clear(self):
        for path in self.root.iterdir():
            path.unlink()
        self.chunks.clear()


class UIAStatistics:
    def __init__(self, directory=None):
        self.real = FeatureBank(directory)
        self.fake = FeatureBank(directory)
        self.real_feature_mean = self.real_inv_covariance = None
        self.fake_feature_mean = self.fake_inv_covariance = None

    def fit(self):
        # Every sample participates, in the original concatenation order.
        # sklearn's estimator and its arithmetic are unchanged; mapped input
        # avoids an additional resident torch.cat allocation.
        for label, bank in (("real", self.real), ("fake", self.fake)):
            values = bank.concatenate()
            precision = torch.Tensor(LedoitWolf().fit(values.cpu()).precision_)
            mean = values.mean(dim=0).cpu()
            setattr(self, f"{label}_inv_covariance", precision.cpu())
            setattr(self, f"{label}_feature_mean", mean)
            del values
            bank.clear()

    def collect(self, patches, labels):
        real = torch.where(labels == 0.0)[0]
        features = patches[real[:4]]
        self.real.append(features.reshape(-1, 768).cpu().detach())
        fake = torch.where(labels == 1.0)[0]
        features = patches[fake[:4], 3:11, 3:11, :]
        self.fake.append(features.reshape(-1, 768).cpu().detach())

    def state_dict(self):
        return {
            "real_feature_list": self.real.tensors(),
            "fake_feature_list": self.fake.tensors(),
            **{
                name: getattr(self, name)
                for name in (
                    "real_feature_mean",
                    "real_inv_covariance",
                    "fake_feature_mean",
                    "fake_inv_covariance",
                )
            },
        }

    def load_state_dict(self, state):
        expected = {
            "real_feature_list",
            "fake_feature_list",
            "real_feature_mean",
            "real_inv_covariance",
            "fake_feature_mean",
            "fake_inv_covariance",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("UIA requires complete banks, means and covariance state")
        for label, bank in (("real", self.real), ("fake", self.fake)):
            bank.clear()
            chunks = state[f"{label}_feature_list"]
            if not isinstance(chunks, list):
                raise ValueError("UIA banks must be ordered tensor lists")
            for chunk in chunks:
                if not torch.isfinite(chunk).all():
                    raise ValueError("UIA banks contain nonfinite features")
                bank.append(chunk)
            mean, precision = (
                state[f"{label}_feature_mean"],
                state[f"{label}_inv_covariance"],
            )
            if (mean is None) != (precision is None):
                raise ValueError("UIA mean and precision must be present together")
            if mean is not None:
                if (
                    tuple(mean.shape) != (768,)
                    or tuple(precision.shape) != (768, 768)
                    or not torch.isfinite(mean).all()
                    or not torch.isfinite(precision).all()
                ):
                    raise ValueError("Invalid UIA covariance geometry or values")
            setattr(self, f"{label}_feature_mean", mean)
            setattr(self, f"{label}_inv_covariance", precision)
