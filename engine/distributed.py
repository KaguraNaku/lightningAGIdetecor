"""Single-host torchrun workers with CPU collectives."""

from contextlib import contextmanager
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist

from .accelerator import resolve_device


@dataclass(frozen=True)
class DistributedRun:
    device: torch.device
    rank: int
    world_size: int

    @classmethod
    @contextmanager
    def launch(cls, request):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size == 1:
            yield None
            return
        if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size:
            raise ValueError("Distributed runs require all torchrun workers on a single host")
        device = resolve_device(request)
        if device.type != "cpu":
            index = int(os.environ["LOCAL_RANK"])
            device = resolve_device(torch.device(device.type, index))
            torch.accelerator.set_device_index(device)

        dist.init_process_group("gloo")
        try:
            yield cls(device, dist.get_rank(), world_size)
        finally:
            dist.destroy_process_group()

    @property
    def is_main(self):
        return self.rank == 0

    def broadcast(self, value):
        objects = [value]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def gather(self, value):
        objects = [None] * self.world_size if self.is_main else None
        dist.gather_object(value, objects, dst=0)
        return objects

    def sum_array(self, array):
        """Sum a contiguous NumPy array in place on rank zero."""
        dist.reduce(torch.from_numpy(array).view(-1), dst=0, op=dist.ReduceOp.SUM)

    def barrier(self):
        dist.barrier()
