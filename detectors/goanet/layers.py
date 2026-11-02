"""Native SVD tail projection with a model-owned, scoped prior context."""

from contextlib import contextmanager
import torch
from torch import nn


class PriorContext:
    def __init__(self):
        self.value = None

    @contextmanager
    def bind(self, value):
        previous = self.value
        self.value = value
        try:
            yield
        finally:
            self.value = previous


class PriorGuidedGOALayer(nn.Module):
    def __init__(self, linear, context):
        super().__init__()
        weight = linear.weight.data.clone()
        size = min(weight.shape)
        residual = min(32, size)
        principal = size - residual
        if weight.is_meta:
            # Restoration needs only native shapes; saved tensors supply all
            # SVD values. No approximation is used for real initialization.
            u = torch.empty(
                weight.shape[0], size, device=weight.device, dtype=weight.dtype
            )
            singular = torch.empty(size, device=weight.device, dtype=weight.dtype)
            vh = torch.empty(
                size, weight.shape[1], device=weight.device, dtype=weight.dtype
            )
        else:
            u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        self.U_pri = nn.Parameter(u[:, :principal], requires_grad=False)
        self.S_pri = nn.Parameter(singular[:principal], requires_grad=False)
        self.V_pri = nn.Parameter(vh[:principal, :], requires_grad=False)
        self.S_res = nn.Parameter(singular[principal:].clone())
        self.U_res = nn.Parameter(u[:, principal:].clone())
        self.V_res = nn.Parameter(vh[principal:, :].clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)
        self.prior_gate = nn.Sequential(nn.Linear(64, residual), nn.Tanh())
        nn.init.zeros_(self.prior_gate[0].weight)
        nn.init.zeros_(self.prior_gate[0].bias)
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.k_pri, self.k_res = principal, residual
        self.context = context

    def forward(self, values):
        principal = (values @ self.V_pri.t()) * self.S_pri.unsqueeze(0).unsqueeze(0)
        principal = principal @ self.U_pri.t()
        projection = self.U_pri @ (self.U_pri.t() @ self.U_res)
        orthogonal = self.U_res - projection
        prior = self.context.value
        if prior is not None:
            gate = self.prior_gate(prior)
            modulated = self.S_res.unsqueeze(0) * (1.0 + 0.5 * gate)
        residual = values @ self.V_res.t()
        if prior is not None:
            residual = residual * modulated.unsqueeze(1)
        else:
            residual = residual * self.S_res.unsqueeze(0).unsqueeze(0)
        residual = residual @ orthogonal.t()
        output = principal + self.res_scale * residual
        return output + self.bias if self.bias is not None else output

    def compute_orthogonal_loss(self):
        return torch.norm(self.U_pri.t() @ self.U_res, p="fro") ** 2


def replace_attention(model, context):
    def replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, PriorGuidedGOALayer(child, context))
            else:
                replace(child)

    def visit(module):
        for name, child in module.named_children():
            if "self_attn" in name or "attn" in name:
                replace(child)
            else:
                visit(child)

    visit(model)
    for name, p in model.named_parameters():
        p.requires_grad_(
            any(
                key in name
                for key in ("U_res", "S_res", "V_res", "prior_gate", "res_scale")
            )
        )
    return model
