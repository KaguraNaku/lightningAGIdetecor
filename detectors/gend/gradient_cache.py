"""Exact logical-batch GenD loss with bounded encoder activation storage.

Only model execution is split. The objective still sees every pair in the
logical batch, including pairs spanning execution chunks. Augmented input
tensors are reused, and the optimizer remains owned by the shared trainer.
"""

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn

from engine.accelerator import get_device_rng_state, set_device_rng_state

from .detector import GenDDetector
from .encoder import GenDCLIPEncoder


@dataclass
class ForwardRandomState:
    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    accelerator: torch.Tensor | None
    device: torch.device

    @classmethod
    def capture(cls, device):
        accelerator = get_device_rng_state(device) if device.type != "cpu" else None
        return cls(
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            accelerator,
            device,
        )

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch_cpu)
        if self.accelerator is not None:
            set_device_rng_state(self.device, self.accelerator)


class GenDGradientCache:
    """Replay GenD outputs using their full-batch upstream gradients.

    This is deliberately restricted to the sample-independent, zero-dropout GenD
    CLIP and linear head. It is not a generic promise that splitting BatchNorm or
    another batch-dependent model preserves its training protocol.
    """

    ALGORITHM = "gend-logical-batch-output-gradient-cache-v1"
    OUTPUTS = ("cls", "l2_embeddings")

    def __init__(self, model, microbatch_size):
        if type(microbatch_size) is not int or microbatch_size < 1:
            raise ValueError(
                "gradient_cache_microbatch_size must be a positive integer"
            )
        if (
            type(model) is not GenDDetector
            or type(model.model.feature_extractor) is not GenDCLIPEncoder
        ):
            raise ValueError(
                "Gradient caching requires the native sample-independent GenD CLIP model"
            )
        if "forward_outputs" in model.__dict__:
            raise ValueError("Gradient caching refuses replaced GenD forward methods")
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise ValueError("Gradient caching cannot split BatchNorm semantics")
            if (
                isinstance(
                    module,
                    (
                        nn.Dropout,
                        nn.Dropout1d,
                        nn.Dropout2d,
                        nn.Dropout3d,
                        nn.AlphaDropout,
                        nn.FeatureAlphaDropout,
                    ),
                )
                and module.p
            ):
                raise ValueError(
                    "Gradient caching requires the native zero-dropout GenD model"
                )
            config = getattr(module, "config", None)
            if config is not None and getattr(config, "attention_dropout", 0.0):
                raise ValueError("Gradient caching requires zero attention dropout")
        if any(not name.endswith(".position_ids") for name, _ in model.named_buffers()):
            raise ValueError(
                "Gradient caching refuses model buffers with unknown update semantics"
            )
        self.model, self.microbatch_size = model, microbatch_size
        self.cached_images = self.replayed_images = 0
        self.cache_forwards = self.replay_forwards = 0

    def policy(self, logical_batch_size):
        return {
            "algorithm": self.ALGORITHM,
            "microbatch_size": self.microbatch_size,
            "logical_batch_size": logical_batch_size,
            "objective_scope": "all logical-batch CE examples and all label/alignment/uniformity pairs",
            "optimizer_steps": "one per logical batch;no extra loss division by chunks",
            "data_augmentation": "once in loader;reuse identical already augmented tensors",
            "model_execution": "one no-grad cache pass and one gradient replay pass",
            "sample_exposures": "logical samples once;model forward images separately count both passes",
        }

    def _buffers(self):
        return {
            name: (id(buffer), buffer._version)
            for name, buffer in self.model.named_buffers()
        }

    def backward(self, images, labels, *, objective, scaler, runtime, validate_loss):
        if runtime.precision != "fp32":
            raise ValueError(
                "GenD gradient caching currently requires its native FP32 protocol"
            )
        if not self.model.training:
            raise ValueError("GenD gradient caching is a training operation")
        if len(images) != len(labels) or len(labels) < 2:
            raise ValueError("GenD needs at least two aligned logical-batch samples")
        chunks = [
            slice(start, min(start + self.microbatch_size, len(labels)))
            for start in range(0, len(labels), self.microbatch_size)
        ]
        states, parts = [], {name: [] for name in self.OUTPUTS}
        buffers = self._buffers()
        with torch.no_grad():
            for chunk in chunks:
                states.append(ForwardRandomState.capture(runtime.device))
                outputs = self.model.forward_outputs(images[chunk])
                if self._buffers() != buffers:
                    raise ValueError(
                        "GenD forward updated model buffers during gradient caching"
                    )
                for name in self.OUTPUTS:
                    parts[name].append(outputs[name])
                self.cached_images += len(labels[chunk])
                self.cache_forwards += 1
        after_cache = ForwardRandomState.capture(runtime.device)
        cached = {
            name: torch.cat(parts[name]).detach().requires_grad_(True)
            for name in self.OUTPUTS
        }
        del parts, outputs
        loss = objective(cached, labels)
        validate_loss(loss)
        upstream = torch.autograd.grad(scaler.scale(loss), tuple(cached.values()))
        try:
            for chunk, state in zip(chunks, states, strict=True):
                state.restore()
                outputs = self.model.forward_outputs(images[chunk])
                if self._buffers() != buffers:
                    raise ValueError(
                        "GenD forward updated model buffers during gradient replay"
                    )
                self.replayed_images += len(labels[chunk])
                self.replay_forwards += 1
                torch.autograd.backward(
                    tuple(outputs[name] for name in self.OUTPUTS),
                    tuple(gradient[chunk] for gradient in upstream),
                )
                del outputs
        finally:
            # Recomputing activations must not advance the experiment RNG twice.
            after_cache.restore()
        return loss.detach()
