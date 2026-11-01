"""SVD residual parameterization from the pinned Effort author implementation.

Author commit: 96f5dea2b534d400cfd7003f053c7e93c8e16461.
Retain the frozen leading components as checkpoint parameters for strict legacy
compatibility. The published experiment's active objective is cross entropy.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, retained_rank, *, bias=True):
        super().__init__()
        rank = min(in_features, out_features)
        if type(retained_rank) is not int or not 0 <= retained_rank < rank:
            raise ValueError("retained_rank must leave at least one residual component")
        self.in_features, self.out_features = in_features, out_features
        self.r = retained_rank
        residual_rank = rank - retained_rank
        self.weight_main = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        )
        self.U_residual = nn.Parameter(torch.empty(out_features, residual_rank))
        self.S_residual = nn.Parameter(torch.empty(residual_rank))
        self.V_residual = nn.Parameter(torch.empty(residual_rank, in_features))
        self.U_r = nn.Parameter(torch.empty(out_features, retained_rank), requires_grad=False)
        self.S_r = nn.Parameter(torch.empty(retained_rank), requires_grad=False)
        self.V_r = nn.Parameter(torch.empty(retained_rank, in_features), requires_grad=False)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, retained_rank, *, initialize=True):
        with torch.device(linear.weight.device):
            module = cls(
                linear.in_features, linear.out_features, retained_rank,
                bias=linear.bias is not None,
            )
        module.to(dtype=linear.weight.dtype)
        if initialize:
            # Keep the author's operation order, including explicit diagonal matrices.
            u, s, vh = torch.linalg.svd(linear.weight.detach().float(), full_matrices=False)
            r = retained_rank
            module.U_r.copy_(u[:, :r])
            module.S_r.copy_(s[:r])
            module.V_r.copy_(vh[:r])
            module.weight_main.copy_(u[:, :r] @ torch.diag(s[:r]) @ vh[:r])
            module.U_residual.copy_(u[:, r:])
            module.S_residual.copy_(s[r:])
            module.V_residual.copy_(vh[r:])
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def compute_current_weight(self):
        return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual

    def forward(self, inputs):
        return F.linear(inputs, self.compute_current_weight(), self.bias)


def replace_attention(vision, retained_rank, *, initialize=True):
    """Replace q/k/v/out projections only; all other backbone parameters stay frozen."""
    vision.requires_grad_(False)
    for layer in vision.encoder.layers:
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear = getattr(layer.self_attn, name)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"Expected an unmodified attention Linear at {name}")
            setattr(
                layer.self_attn, name,
                SVDResidualLinear.from_linear(linear, retained_rank, initialize=initialize),
            )
    return vision
