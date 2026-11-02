"""Runtime resources and artifacts; no detector-specific imports."""

from contextlib import nullcontext
from functools import partial
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import uuid

import numpy as np
import torch

from .task import ExperimentTask


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    precision: str = "fp32"

    @classmethod
    def create(cls, settings, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(settings["device"])
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu and cuda")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError("CUDA device index is out of range")
            if settings["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("CUDA device does not support bf16")
        if device.type == "cpu" and settings["precision"] == "fp16":
            raise ValueError("fp16 requires CUDA; use fp32 or bf16 on CPU")
        if "allow_tf32" in settings:
            torch.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
            torch.backends.cudnn.allow_tf32 = settings["allow_tf32"]
        return cls(device, settings["precision"])

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(self.device.type, dtype=dtype)


def create_task(model, *, state=None) -> ExperimentTask:
    module, name = model["factory"].split(":", 1)
    factory = getattr(import_module(module), name)
    settings = {k: v for k, v in model.items() if k != "factory"}
    task = factory(settings, state=state)
    for method in ("fit", "predict", "state_dict"):
        if not callable(getattr(task, method, None)):
            raise TypeError(f"{model['factory']} must provide callable {method}")
    if not any(
        callable(getattr(task, name, None)) for name in ("get_collate_fn", "collate_fn")
    ):
        raise TypeError("Task must provide get_collate_fn(stage) or collate_fn")
    return task


def collator_for(task, stage):
    if stage not in ("train", "validation", "evaluation"):
        raise ValueError(f"Unknown preprocessing stage: {stage}")
    provider = getattr(task, "get_collate_fn", None)
    collator = provider(stage) if callable(provider) else task.collate_fn
    if not callable(collator):
        raise TypeError(f"Task returned a noncallable collator for {stage}")
    return collator


def dataset_adapter_for(task, stage):
    """Optionally adapt a resolved dataset before sampling or worker creation."""
    if stage not in ("train", "validation", "evaluation"):
        raise ValueError(f"Unknown dataset stage: {stage}")
    provider = getattr(task, "prepare_dataset", None)
    if provider is None:
        return None
    if not callable(provider):
        raise TypeError("Task prepare_dataset must be callable")
    return partial(provider, stage=stage)


def create_run_dir(root, command):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = (
        Path(root).expanduser().resolve() / f"{stamp}-{command}-{uuid.uuid4().hex[:8]}"
    )
    path.mkdir(parents=True)
    return path


def configure_logging(directory, level):
    """Return owned handlers so repeated programmatic runs do not leak handlers."""
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(level)
    handlers = [logging.StreamHandler(), logging.FileHandler(directory / "run.log")]
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root.addHandler(handler)
    return root, handlers, previous_level


def release_cpu_memory():
    """Release unreachable models and free allocator pages after a large handoff."""
    import ctypes
    import gc

    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)


def atomic_save(path, value, *, torch_format=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    os.close(fd)
    try:
        if torch_format:
            with open(temporary, "wb") as stream:
                torch.save(value, stream)
        else:
            Path(temporary).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
