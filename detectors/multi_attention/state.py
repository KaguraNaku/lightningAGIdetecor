"""Explicit state formerly absent from the native detector's raw state_dict."""

import math

import torch


def pack_state(model, objective, scheduler=None):
    return {
        "forward_batches": model.batch_cnt,
        "ril": {
            "alpha": objective.ril.alpha,
            "batch_cnt": objective.ril.batch_cnt,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in objective.ril.state_dict().items()
            },
        },
        "scheduler": scheduler,
    }


def restore_state(model, objective, state, *, assign=False):
    if not isinstance(state, dict) or set(state) != {
        "forward_batches",
        "ril",
        "scheduler",
    }:
        raise ValueError(
            "MultiAttention requires complete forward/RIL/scheduler method state"
        )
    ril = state["ril"]
    if not isinstance(ril, dict) or set(ril) != {"alpha", "batch_cnt", "state_dict"}:
        raise ValueError("Incomplete MultiAttention RIL state")
    for value in (state["forward_batches"], ril["batch_cnt"]):
        if type(value) is not int or value < 0:
            raise ValueError(
                "MultiAttention batch counters must be nonnegative integers"
            )
    alpha = ril["alpha"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("MultiAttention RIL alpha must be positive and finite")
    if state["scheduler"] is not None and not isinstance(state["scheduler"], dict):
        raise ValueError("Invalid MultiAttention scheduler state")
    objective.ril.load_state_dict(ril["state_dict"], strict=True, assign=assign)
    if any(not torch.isfinite(value).all() for value in objective.ril.buffers()):
        raise ValueError("MultiAttention RIL buffers must be finite")
    model.batch_cnt = state["forward_batches"]
    objective.ril.alpha, objective.ril.batch_cnt = float(alpha), ril["batch_cnt"]
    return state["scheduler"]
