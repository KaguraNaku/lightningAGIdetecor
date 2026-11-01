"""LoRA attention projections with a canonical unmerged checkpoint layout.

The train and eval algebra matches loralib's Linear, but eval never changes
the stored base weight. Repeated mode switches therefore cannot corrupt state.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, *, rank, alpha, bias=True):
        super().__init__()
        if type(rank) is not int or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("LoRA alpha must be finite and positive")
        self.in_features, self.out_features = in_features, out_features
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=False)
            if bias
            else None
        )
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        # loralib initializes its base Linear twice before copying pretrained
        # weights. Retain those RNG draws so adapter initialization is seed-aligned.
        for _ in range(2):
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                bound = 1 / math.sqrt(in_features)
                nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, *, rank, alpha):
        with torch.device(linear.weight.device):
            layer = cls(
                linear.in_features,
                linear.out_features,
                rank=rank,
                alpha=alpha,
                bias=linear.bias is not None,
            )
        layer.to(dtype=linear.weight.dtype)
        layer.weight.copy_(linear.weight)
        if layer.bias is not None:
            layer.bias.copy_(linear.bias)
        return layer

    def delta_weight(self):
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, inputs):
        if self.training:
            output = F.linear(inputs, self.weight, self.bias)
            return output + (inputs @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return F.linear(inputs, self.weight + self.delta_weight(), self.bias)


def replace_attention(vision, *, rank, alpha):
    vision.requires_grad_(False)
    for block in vision.encoder.layers:
        projections = [
            (name, layer)
            for name, layer in block.self_attn.named_children()
            if name in ("q_proj", "k_proj", "v_proj", "out_proj")
        ]
        if len(projections) != 4:
            raise ValueError("CLIP attention must have four linear projections")
        for name, original in projections:
            if not isinstance(original, nn.Linear):
                raise TypeError("Expected an unmodified CLIP attention projection")
            setattr(
                block.self_attn,
                name,
                LoRALinear.from_linear(original, rank=rank, alpha=alpha),
            )
    return vision
