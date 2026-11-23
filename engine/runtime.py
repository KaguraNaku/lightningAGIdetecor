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
from typing import TYPE_CHECKING
import uuid

import numpy as np
import torch

from .accelerator import resolve_device, set_tf32, validate_precision
from .gpu_timing import GPUTimeRecorder
from .task import ExperimentTask

if TYPE_CHECKING:
    from .distributed import DistributedRun


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    precision: str = "fp32"
    timer: GPUTimeRecorder | None = None
    group: "DistributedRun | None" = None

    @classmethod
    def create(cls, settings, seed, *, timing=False, group=None):
        device = group.device if group is not None else resolve_device(settings["device"])
        precision = settings["precision"]
        validate_precision(device, precision)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda" and "allow_tf32" in settings:
            set_tf32(settings["allow_tf32"], settings["allow_tf32"])
        timer = GPUTimeRecorder(device=device) if timing else None
        return cls(device, precision, timer, group)

    def grad_scaler(self):
        if self.precision == "fp16" and self.device.type == "mps":
            raise ValueError("Use fp32 or bf16 for training on MPS")
        return torch.amp.GradScaler(self.device.type, enabled=self.precision == "fp16")

    def time_region(self, phase, *, samples=None):
        if self.timer is None:
            return nullcontext()
        if samples is None:
            return self.timer.region(phase, is_batch=False, measure_device=False)
        return self.timer.region(phase, samples=samples)

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(self.device.type, dtype=dtype)


def task_factory(model):
    from .models import ModelRegistry

    module, name = ModelRegistry.resolve_factory(model["factory"]).split(":", 1)
    return getattr(import_module(module), name)


def create_task(model, *, state=None) -> ExperimentTask:
    factory = task_factory(model)
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


def sampler_factory_for(task, stage):
    """Optionally construct a method sampler after dataset adaptation."""
    if stage not in ("train", "validation", "evaluation"):
        raise ValueError(f"Unknown sampler stage: {stage}")
    provider = getattr(task, "get_sampler", None)
    if provider is None:
        return None
    if not callable(provider):
        raise TypeError("Task get_sampler must be callable")
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


class _CheckpointWriter:
    """Bound dirty file-cache growth while torch serializes a large checkpoint."""

    def __init__(self, stream, chunk_bytes=8 * 1024**2):
        self.stream, self.chunk_bytes, self.pending = stream, chunk_bytes, 0

    def write(self, data):
        data = memoryview(data)
        total = len(data)
        for start in range(0, total, self.chunk_bytes):
            block = data[start : start + self.chunk_bytes]
            written = self.stream.write(block)
            if written != len(block):
                raise OSError("Incomplete checkpoint write")
            self.pending += written
            if self.pending >= self.chunk_bytes:
                self.flush()
        return total

    def flush(self):
        self.stream.flush()
        if self.pending:
            getattr(os, "fdatasync", os.fsync)(self.stream.fileno())
            advise = getattr(os, "posix_fadvise", None)
            advice = getattr(os, "POSIX_FADV_DONTNEED", None)
            if advise is not None and advice is not None:
                advise(self.stream.fileno(), 0, 0, advice)
            self.pending = 0

    def tell(self):
        return self.stream.tell()


def atomic_save(path, value, *, torch_format=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    os.close(fd)
    try:
        if torch_format:
            with open(temporary, "wb") as stream:
                torch.save(value, _CheckpointWriter(stream))
        else:
            Path(temporary).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n"
            )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
