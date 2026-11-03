"""Shared frozen-module and patch-readout behavior; no model-class imports."""

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


class FrozenVisionEncoder(nn.Module):
    """A configured per-block readout with native weights registered once.

    forward returns (N, layers, channels); readouts returns all four feature
    views from a single backbone pass. Mean is L2-normalized; std is the raw
    sqrt(clamp(E[x²]-E[x]²,min=1e-6)); mean_std concatenates these two vectors.
    This matches the frozen experiments and does not redefine native pooling.
    """

    MODEL_TYPE = None
    PREFIX_TOKENS = 0
    VERSION = 1

    def __init__(self, vision_model, model_id=None, *, readout="global", revision=None):
        super().__init__()
        if vision_model.config.model_type != self.MODEL_TYPE:
            raise ValueError(f"{type(self).__name__} requires {self.MODEL_TYPE}")
        if readout not in ("global", "mean", "std", "mean_std"):
            raise ValueError("unknown vision readout")
        self.model = vision_model
        self.model_id = str(
            model_id or vision_model.config._name_or_path or self.MODEL_TYPE
        )
        self.readout = readout
        self.revision = getattr(vision_model.config, "_commit_hash", None) or revision
        self.model.requires_grad_(False)
        self.eval()

    @property
    def feature_id(self):
        """Include readout/revision so ridge statistics cannot cross adapters."""
        value = {
            "version": self.VERSION,
            "model_type": self.MODEL_TYPE,
            "model_id": self.model_id,
            "readout": self.readout,
            "revision": self.revision,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def get_extra_state(self):
        return {
            "version": self.VERSION,
            "model_type": self.MODEL_TYPE,
            "model_id": self.model_id,
            "readout": self.readout,
            "revision": self.revision,
            "feature_id": self.feature_id,
        }

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise ValueError(
                "saved encoder identity/readout/revision does not match configuration"
            )

    def train(self, mode=True):
        return super().train(False)

    def _global(self, block):
        raise NotImplementedError

    def _model_kwargs(self):
        return {}

    def _validate_pixel_values(self, pixel_values):
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise ValueError("pixel_values must be a processed NCHW tensor")

    @torch.no_grad()
    def _extract(self, pixel_values, requested):
        self._validate_pixel_values(pixel_values)
        parameter = next(self.model.parameters())
        outputs = self.model(
            pixel_values=pixel_values.to(parameter),
            output_hidden_states=True,
            return_dict=True,
            **self._model_kwargs(),
        )
        hidden = outputs.hidden_states[1:]
        if len(hidden) != self.model.config.num_hidden_layers:
            raise ValueError(
                "backbone did not return one hidden state per transformer block"
            )
        values = {name: [] for name in requested}
        spatial = tuple(name for name in requested if name != "global")
        for block in hidden:
            if "global" in values:
                values["global"].append(F.normalize(self._global(block), dim=-1))
            if spatial:
                patches = block[:, self.PREFIX_TOKENS :]
                mean = patches.mean(1)
                if "mean" in spatial or "mean_std" in spatial:
                    normalized_mean = F.normalize(mean, dim=-1)
                if "std" in spatial or "mean_std" in spatial:
                    std = (
                        (patches.square().mean(1) - mean.square())
                        .clamp(min=1e-6)
                        .sqrt()
                    )
                if "mean" in spatial:
                    values["mean"].append(normalized_mean)
                if "std" in spatial:
                    values["std"].append(std)
                if "mean_std" in spatial:
                    values["mean_std"].append(torch.cat((normalized_mean, std), dim=-1))
        return {name: torch.stack(rows, dim=1) for name, rows in values.items()}

    def forward(self, pixel_values):
        return self._extract(pixel_values, (self.readout,))[self.readout]

    def readouts(self, pixel_values):
        """Compute global, mean, std and mean_std in one frozen model pass."""
        return self._extract(pixel_values, ("global", "mean", "std", "mean_std"))
