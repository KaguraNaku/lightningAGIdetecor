"""Selected VideoMAE-base encoder, preserving the local HF4.57 state/math.

The explicit CPU/fp32 eager profile is independently auditable. Published
pretraining decoder/masking and optional attention backends are not selected by
the native deepfake detector. Full encoder numeric validation is still pending.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def position_table():
    # Keep the native float64 NumPy calculation followed by FloatTensor cast.
    # A torch/vectorized alternative can round the fixed positions differently.
    table = np.array([
        [position / np.power(10000, 2 * (dimension // 2) / 768)
         for dimension in range(768)]
        for position in range(1568)
    ])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table).unsqueeze(0)


class PatchEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(3, 768, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, clips):
        return self.projection(clips.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings()
        # Native positions are a plain nonpersistent tensor, not learned state.
        self.position_embeddings = position_table()

    def forward(self, clips):
        value = self.patch_embeddings(clips)
        return value + self.position_embeddings.detach().type_as(value).to(
            device=value.device, copy=True
        )


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768, bias=False)
        self.key = nn.Linear(768, 768, bias=False)
        self.value = nn.Linear(768, 768, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(768))
        self.v_bias = nn.Parameter(torch.zeros(768))
        self.scaling = 64**-0.5

    def forward(self, value):
        batch, count, _ = value.shape
        # Preserve the explicit zero key bias and separate q/v parameters from
        # legacy HF. Current HF's Linear.bias layout is not interchangeable.
        keys = F.linear(value, self.key.weight, torch.zeros_like(self.v_bias, requires_grad=False))
        values = F.linear(value, self.value.weight, self.v_bias)
        queries = F.linear(value, self.query.weight, self.q_bias)
        key, content, query = (
            part.view(batch, count, 12, 64).transpose(1, 2)
            for part in (keys, values, queries)
        )
        weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=0.0, training=self.training)
        result = torch.matmul(weights, content).transpose(1, 2).contiguous()
        return result.reshape(batch, count, 768)


class SelfOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.0)

    def forward(self, value):
        return self.dropout(self.dense(value))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.output = SelfOutput()

    def forward(self, value):
        return self.output(self.attention(value))


class Intermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 3072)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, value):
        return self.intermediate_act_fn(self.dense(value))


class Output(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        self.dropout = nn.Dropout(0.0)

    def forward(self, value, residual):
        return self.dropout(self.dense(value)) + residual


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.intermediate = Intermediate()
        self.output = Output()
        self.layernorm_before = nn.LayerNorm(768, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(768, eps=1e-12)

    def forward(self, value):
        value = self.attention(self.layernorm_before(value)) + value
        return self.output(self.intermediate(self.layernorm_after(value)), value)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([Layer() for _ in range(12)])

    def forward(self, value):
        for layer in self.layer:
            value = layer(value)
        return value


class VideoMAEBackbone(nn.Module):
    """Fixed published-base geometry and native eager inference/training math."""

    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.encoder = Encoder()
        # The published config has use_mean_pooling=False. The detector's own
        # mean and second LayerNorm occur AFTER this internal token LayerNorm.
        self.layernorm = nn.LayerNorm(768, eps=1e-12)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (16, 3, 224, 224):
            raise ValueError("VideoMAE-base needs NTCHW RGB224/T16 clips")
        return self.layernorm(self.encoder(self.embeddings(clips)))
