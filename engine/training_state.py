"""Save selected training state and restore it at epoch boundaries."""
import random
from uuid import uuid4

import numpy as np
import torch

from .fit import read_checkpoint


def qualified(value):
    return type(value).__module__ + '.' + type(value).__qualname__


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
    provider = getattr(sampler, 'state_dict', None)
    if callable(provider):
        state['sampler'] = provider()
    generator = getattr(sampler, 'generator', None)
    if isinstance(generator, torch.Generator):
        state['sampler_generator'] = generator.get_state()
    return state


def _restore_generator(generator, saved):
    if isinstance(generator, torch.Generator) != (saved is not None):
        raise ValueError("Checkpoint loader or sampler generator configuration differs")
    if saved is not None:
        generator.set_state(saved)


def restore_loader(loader, state):
    if qualified(loader.sampler) != state['type']:
        raise ValueError('Checkpoint sampler type differs')
    if loader.num_workers != state['num_workers'] or loader.batch_size != state['batch_size']:
        raise ValueError('Checkpoint loader batch size or worker count differs')
    if loader.persistent_workers or state.get("persistent_workers", False):
        raise ValueError("Checkpoint resume requires nonpersistent workers")
    _restore_generator(loader.generator, state["generator"])
    if 'sampler' in state:
        loader.sampler.load_state_dict(state['sampler'])
    _restore_generator(getattr(loader.sampler, "generator", None), state.get("sampler_generator"))


def optimizer_names(trainer, optimizer):
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def restore_training_state(saved, trainer, optimizer, scaler, loader, validation_loader, runtime):
    contract = saved['training_state']
    if (contract["optimizer_type"] != qualified(optimizer)
            or contract["optimizer_parameter_names"] != optimizer_names(trainer, optimizer)):
        raise ValueError("Checkpoint optimizer type or parameter layout differs")
    before = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
    carried = [{k: v for k, v in group.items() if k != "params"}
               for group in saved["optimizer"]["param_groups"]]
    check_contract = getattr(trainer, "validate_optimizer_contract", None)
    if check_contract is not None:
        check_contract(before, carried)
    elif before != carried:
        raise ValueError("Checkpoint optimizer hyperparameters differ")
    load_weights = getattr(trainer, "load_selected_weights", trainer.model.load_state_dict)
    load_weights(saved["state_dict"])
    if (objective := getattr(trainer, "objective", None)) is not None:
        objective.load_state_dict(saved["objective"])
    if (load_method := getattr(trainer, "load_method_state_dict", None)) is not None:
        load_method(saved.get("method_state"))
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    restore_loader(loader, contract["loaders"]["train"])
    restore_loader(validation_loader, contract["loaders"]["validation"])
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    kind, state, *tail = saved["numpy_rng"]
    np.random.set_state((kind, np.asarray(state, dtype=np.uint32), *tail))
    if runtime.device.type == "cuda":
        if len(saved["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


class TrainingSnapshot:
    def __init__(self, settings, loader, validation_loader, runtime):
        self.id = uuid4().hex
        self.loader, self.validation_loader, self.runtime = loader, validation_loader, runtime
        self.resumable = not (loader.persistent_workers or validation_loader.persistent_workers)
        self.parent = None
        if reference := settings.get("resume_state"):
            self.parent = read_checkpoint(reference["path"])
            state = self.parent.get("training_state", {})
            if state.get("id") != reference["id"] or self.parent["epoch"] != reference["epoch"]:
                raise ValueError("best.pth does not belong to the selected model.pth")
            if (state["device_type"], state["precision"]) != (runtime.device.type, runtime.precision):
                raise ValueError("Resume with the checkpoint's device type and precision")

    def restore(self, trainer, optimizer, scaler):
        if self.parent is not None:
            restore_training_state(
                self.parent, trainer, optimizer, scaler,
                self.loader, self.validation_loader, self.runtime,
            )
            self.parent = None

    def capture(self, trainer, optimizer, scaler):
        numpy_rng = np.random.get_state()
        train = loader_state(self.loader)
        state = {
            "state_dict": trainer.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
            "cuda_rng": torch.cuda.get_rng_state_all() if self.runtime.device.type == "cuda" else [],
            "training_state": {
                "id": self.id,
                "optimizer_type": qualified(optimizer),
                "optimizer_parameter_names": optimizer_names(trainer, optimizer),
                "loaders": {"train": train, "validation": loader_state(self.validation_loader)},
                "device_type": self.runtime.device.type,
                "precision": self.runtime.precision,
            },
        }
        if (objective := getattr(trainer, "objective", None)) is not None:
            state["objective"] = objective.state_dict()
        if (method_state := getattr(trainer, "method_state_dict", None)) is not None:
            state["method_state"] = method_state()
        return state
