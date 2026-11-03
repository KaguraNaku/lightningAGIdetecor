"""Independent native FTCN stop5 temporal head; literal einsum attention."""

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, value, **kwargs):
        return self.fn(value, **kwargs) + value


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, value, **kwargs):
        return self.fn(self.norm(value), **kwargs)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
        )

    def forward(self, value):
        return self.net(value)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 16
        self.scale = 64**-0.5
        self.to_qkv = nn.Linear(1024, 3072, bias=False)
        self.to_out = nn.Sequential(nn.Linear(1024, 1024), nn.Dropout(0.1))

    def forward(self, value, mask=None):
        if mask is not None:
            raise ValueError("The native FTCN TimeTransformer always uses mask=None")
        batch, count, _ = value.shape
        qkv = self.to_qkv(value).chunk(3, dim=-1)
        query, key, content = (
            part.reshape(batch, count, 16, 64).permute(0, 2, 1, 3) for part in qkv
        )
        dots = torch.einsum("b h i d, b h j d -> b h i j", query, key) * self.scale
        attention = dots.softmax(dim=-1)
        result = torch.einsum("b h i j, b h j d -> b h i d", attention, content)
        result = result.permute(0, 2, 1, 3).reshape(batch, count, 1024)
        return self.to_out(result)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(PreNorm(1024, Attention())),
                        Residual(PreNorm(1024, FeedForward())),
                    ]
                )
            ]
        )

    def forward(self, value, mask=None):
        for attention, feed_forward in self.layers:
            value = attention(value, mask=mask)
            value = feed_forward(value)
        return value


class TimeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = 16
        self.pos_embedding = nn.Parameter(torch.randn(1, 17, 1024))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
        self.dropout = nn.Dropout(0.1)
        self.transformer = Transformer()
        self.pool = "cls"
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 1))

    def forward(self, value):
        batch, count, _ = value.shape
        # Match the native einops repeat's expand view, including its gradient
        # reduction, without importing einops into the production pipeline.
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        value = torch.cat((cls_tokens, value), dim=1)
        value += self.pos_embedding[:, : (count + 1)]
        value = self.dropout(value)
        value = self.transformer(value, mask=None)
        return self.mlp_head(self.to_latent(value[:, 0]))


class FTCNTemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d((1, 14, 14))
        self.num_patches = 16
        self.dim = self.in_channels = 1024
        self.time_T = TimeTransformer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, value):
        if value.ndim != 5 or tuple(value.shape[1:]) != (1024, 16, 14, 14):
            raise ValueError("Native FTCN stop5 needs N x1024 x16 x14 x14 features")
        features = self.pool(value)
        tokens = features.reshape(-1, self.in_channels, self.num_patches).permute(
            0, 2, 1
        )
        return self.sigmoid(self.time_T(tokens)), features
