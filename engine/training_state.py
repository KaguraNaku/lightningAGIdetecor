"""Save selected training state and restore it at epoch boundaries."""

import random
from uuid import uuid4

import numpy as np
import torch

from .accelerator import get_device_rng_state, set_device_rng_state
from .fit import read_checkpoint


def qualified(value):
    return type(value).__module__ + "." + type(value).__qualname__


def loader_state(loader):
    sampler = loader.sampler
    generator = loader.generator
    state = {
        "type": qualified(sampler),
        "generator": generator.get_state() if generator is not None else None,
        "num_workers": loader.num_workers,
        "batch_size": loader.batch_size,
        "persistent_workers": loader.persistent_workers,
    }
    provider = getattr(sampler, "state_dict", None)
    if callable(provider):
        state["sampler"] = provider()
    generator = getattr(sampler, "generator", None)
    if isinstance(generator, torch.Generator):
        state["sampler_generator"] = generator.get_state()
    return state


def _restore_generator(generator, saved):
    if isinstance(generator, torch.Generator) != (saved is not None):
        raise ValueError("Checkpoint loader or sampler generator configuration differs")
    if saved is not None:
        generator.set_state(saved)


def restore_loader(loader, state):
    if qualified(loader.sampler) != state["type"]:
        raise ValueError("Checkpoint sampler type differs")
    if (
        loader.num_workers != state["num_workers"]
        or loader.batch_size != state["batch_size"]
    ):
        raise ValueError("Checkpoint loader batch size or worker count differs")
    if loader.persistent_workers or state.get("persistent_workers", False):
        raise ValueError("Checkpoint resume requires nonpersistent workers")
    _restore_generator(loader.generator, state["generator"])
    if "sampler" in state:
        loader.sampler.load_state_dict(state["sampler"])
    _restore_generator(
        getattr(loader.sampler, "generator", None), state.get("sampler_generator")
    )


def optimizer_names(trainer, optimizer):
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def restore_selected_state(trainer, saved):
    trainer.model.load_state_dict(saved["state_dict"])
    if (objective := getattr(trainer, "objective", None)) is not None:
        objective.load_state_dict(saved["objective"])
    method_state = saved.get("method_state")
    if (load_method := getattr(trainer, "load_method_state_dict", None)) is not None:
        load_method(method_state)
    elif method_state is not None:
        raise ValueError("Unexpected method-specific trainer state")


def restore_training_state(
    saved, trainer, optimizer, scaler, loader, validation_loader, runtime
):
    contract = saved["training_state"]
    if contract["optimizer_type"] != qualified(optimizer) or contract[
        "optimizer_parameter_names"
    ] != optimizer_names(trainer, optimizer):
        raise ValueError("Checkpoint optimizer type or parameter layout differs")
    before = [
        {k: v for k, v in group.items() if k != "params"}
        for group in optimizer.param_groups
    ]
    carried = [
        {k: v for k, v in group.items() if k != "params"}
        for group in saved["optimizer"]["param_groups"]
    ]
    check_contract = getattr(trainer, "validate_optimizer_contract", None)
    if check_contract is not None:
        check_contract(before, carried)
    elif before != carried:
        raise ValueError("Checkpoint optimizer hyperparameters differ")
    restore_selected_state(trainer, saved)
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    restore_loader(loader, contract["loaders"]["train"])
    restore_loader(validation_loader, contract["loaders"]["validation"])
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    kind, state, *tail = saved["numpy_rng"]
    np.random.set_state((kind, np.asarray(state, dtype=np.uint32), *tail))
    if runtime.device.type != "cpu":
        states = saved.get("accelerator_rng", saved.get("cuda_rng", []))
        if len(states) != torch.accelerator.device_count():
            raise ValueError("Checkpoint accelerator RNG device count differs")
        for index, state in enumerate(states):
            set_device_rng_state(torch.device(runtime.device.type, index), state)


class TrainingSnapshot:
    def __init__(self, settings, loader, validation_loader, runtime):
        self.id = uuid4().hex
        self.loader, self.validation_loader, self.runtime = (
            loader,
            validation_loader,
            runtime,
        )
        self.resumable = not (
            loader.persistent_workers or validation_loader.persistent_workers
        )
        self.parent = None
        if reference := settings.get("resume_state"):
            self.parent = read_checkpoint(reference["path"])
            state = self.parent.get("training_state", {})
            if (
                state.get("id") != reference["id"]
                or self.parent["epoch"] != reference["epoch"]
            ):
                raise ValueError("best.pth does not belong to the selected model.pth")
            if (state["device_type"], state["precision"]) != (
                runtime.device.type,
                runtime.precision,
            ):
                raise ValueError(
                    "Resume with the checkpoint's device type and precision"
                )

    def restore(self, trainer, optimizer, scaler):
        if self.parent is not None:
            restore_training_state(
                self.parent,
                trainer,
                optimizer,
                scaler,
                self.loader,
                self.validation_loader,
                self.runtime,
            )
            self.parent = None

    def capture(self, trainer, optimizer, scaler):
        numpy_rng = np.random.get_state()
        train = loader_state(self.loader)
        device_type = self.runtime.device.type
        state = {
            "state_dict": trainer.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
            "accelerator_rng": [
                get_device_rng_state(torch.device(device_type, index))
                for index in range(
                    torch.accelerator.device_count() if device_type != "cpu" else 0
                )
            ],
            "training_state": {
                "id": self.id,
                "optimizer_type": qualified(optimizer),
                "optimizer_parameter_names": optimizer_names(trainer, optimizer),
                "loaders": {
                    "train": train,
                    "validation": loader_state(self.validation_loader),
                },
                "device_type": device_type,
                "precision": self.runtime.precision,
            },
        }
        if (objective := getattr(trainer, "objective", None)) is not None:
            state["objective"] = objective.state_dict()
        if (save_method := getattr(trainer, "method_state_dict", None)) is not None:
            state["method_state"] = save_method()
        return state
