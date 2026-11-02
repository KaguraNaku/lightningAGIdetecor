"""Independent native UIA-ViT-B/16 with intermediate raw qk attention."""

import math

import torch
from torch import nn


def native_truncated_normal(tensor):
    """Local timm inverse-CDF sampler for std.02 and bounds[-2,2].

    CDF endpoints are exactly0 and1 for these native preset values. Current
    torch.nn.init.trunc_normal_ uses rejection sampling and consumes different
    random draws, so it cannot replace this seeded recipe.
    """
    if tensor.is_meta:
        return tensor
    with torch.no_grad():
        return (
            tensor.uniform_(-1, 1)
            .erfinv_()
            .mul_(0.02 * math.sqrt(2.0))
            .add_(0.0)
            .clamp_(min=-2.0, max=2.0)
        )


class Mlp(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, features):
        features = self.drop(self.act(self.fc1(features)))
        return self.drop(self.fc2(features))


class Attention(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, features):
        batch, tokens, channels = features.shape
        qkv = (
            self.qkv(features)
            .reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        raw = (query @ key.transpose(-2, -1)) * self.scale
        weights = self.attn_drop(raw.softmax(dim=-1))
        features = (weights @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(features)), raw


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.attn = Attention()
        self.drop_path = nn.Identity()  # Native preset has drop_path_rate=0.
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.mlp = Mlp()

    def forward(self, features):
        attended, raw = self.attn(self.norm1(features))
        features = features + self.drop_path(attended)
        return features + self.drop_path(self.mlp(self.norm2(features))), raw


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)

    def forward(self, images):
        if tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("UIA-ViT requires native RGB224 images")
        return self.proj(images).flatten(2).transpose(1, 2)


class UIAVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        self.pos_drop = nn.Dropout(0.0)
        self.c_real = nn.Parameter(torch.tensor(0.6))
        self.c_fake = nn.Parameter(torch.tensor(0.6))
        self.c_cross = nn.Parameter(torch.tensor(0.2))
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.norm_middle = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(1536, 2)
        native_truncated_normal(self.pos_embed)
        native_truncated_normal(self.cls_token)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            native_truncated_normal(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, images):
        features = self.patch_embed(images)
        features = torch.cat(
            (self.cls_token.expand(images.shape[0], -1, -1), features), dim=1
        )
        features = self.pos_drop(features + self.pos_embed)
        attentions = []
        raw = None
        for index, block in enumerate(self.blocks):
            if index == 6:
                patches = self.norm_middle(features)
            # Source indices8..12 inspect the preceding block's raw logits.
            if index in (8, 9, 10, 11):
                attentions.append(raw)
            features, raw = block(features)
        attentions.append(raw)
        features = self.norm(features)
        patches = patches[:, 1:].reshape(images.shape[0], 14, 14, 768)
        return features, patches, torch.cat(attentions, dim=1)

    def forward(self, images, *, step=1.0):
        features, patches, raw = self.forward_features(images)
        cls, patch_features = features[:, 0], features[:, 1:]
        batch, count, _ = patch_features.shape
        localization = torch.sigmoid(torch.mean(raw[:, :, 0, 1:], dim=1))
        if step < 1 / 8.0:
            localization = (torch.ones(batch, 1, count) / count).to(
                patch_features.device
            )
        else:
            weight = torch.sigmoid(torch.tensor(-12 * (step - 0.7))).to(
                patch_features.device
            )
            localization = (
                weight * torch.ones(batch, 1, count).to(patch_features.device)
                + (1 - weight)
                * localization.reshape(batch, 1, count).to(patch_features.device)
            ) / count
        pooled = torch.cat(
            [cls, torch.bmm(localization, patch_features).squeeze(1)], dim=-1
        )
        return self.head(pooled), patches, raw


def load_uia_vit(model_id, **kwargs):
    if kwargs:
        raise ValueError("UIA-ViT loader preserves the fixed native B/16 preset")
    return UIAVisionTransformer()
