"""Native sidecar attention and globally fused wavelet priors."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .branch import WaveletsEnhancement


class SidecarLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.register_buffer("weight_main", linear.weight.data.clone())
        if linear.bias is None:
            self.bias_main = None
        else:
            self.register_buffer("bias_main", linear.bias.data.clone())
        self.lora_A = nn.Parameter(torch.zeros(32, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, 32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, std=1e-6)

    def forward(self, values):
        dtype = values.dtype
        values = values.to(torch.float32)
        main = F.linear(values, self.weight_main, self.bias_main)
        residual = (values @ self.lora_A.T) @ self.lora_B.T
        return (main + residual * 1.0).to(dtype)

    def compute_orthogonal_loss(self):
        update = self.lora_B @ self.lora_A
        return (
            F.cosine_similarity(
                self.weight_main.view(-1).unsqueeze(0), update.view(-1).unsqueeze(0)
            )
            .abs()
            .squeeze()
        )


def replace_attention(model):
    names = ("q_proj", "k_proj", "v_proj", "out_proj")

    def replace(module):
        if hasattr(module, "self_attn") and isinstance(module.self_attn, nn.ModuleDict):
            for key in names:
                if key in module.self_attn and isinstance(
                    module.self_attn[key], nn.Linear
                ):
                    sidecar = SidecarLinear(module.self_attn[key])
                    module.self_attn[key] = sidecar
                    if hasattr(module, key):
                        setattr(module, key, sidecar)
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in names:
                setattr(module, name, SidecarLinear(child))
            else:
                replace(child)

    replace(model)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name or "wavelets_branch" in name or "head" in name)
    return model


class EffluxletDetector(nn.Module):
    def __init__(self, vision):
        super().__init__()
        # Native constructor creates all wavelet weights before sidecar draws.
        branch = WaveletsEnhancement(vision.config.hidden_size)
        self.backbone = replace_attention(vision)
        self.wavelets_branch = branch
        self.head = nn.Linear(vision.config.hidden_size, 2)

    def forward_outputs(self, images):
        wavelets = self.wavelets_branch(images)
        features = self.backbone(images).pooler_output
        if wavelets:
            contexts = [
                v.flatten(1, -2).mean(1) if v.ndim > 2 else v
                for v in wavelets
                if isinstance(v, torch.Tensor)
            ]
            if contexts:
                features = features + torch.stack(contexts).sum(0)
        logits = self.head(features)
        return {"cls": logits, "prob": logits.softmax(1)[:, 1], "feat": features}

    def forward(self, images):
        return self.forward_outputs(images)["cls"]

    def loss_terms(self, outputs, labels):
        classification = F.cross_entropy(outputs["cls"], labels)
        penalties = [
            m.compute_orthogonal_loss()
            for m in self.backbone.modules()
            if isinstance(m, SidecarLinear)
        ]
        orthogonal = (
            torch.mean(torch.stack(penalties)) * 100.0
            if penalties
            else torch.tensor(0.0).to(outputs["cls"].device)
        )
        return {
            "overall": classification + orthogonal,
            "cls_loss": classification,
            "ortho_loss": orthogonal,
        }
