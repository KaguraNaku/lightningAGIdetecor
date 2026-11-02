"""Native stateful warmup/ensemble forward and constructor BN probe."""

import torch
from torch import nn
from torch.nn import functional as F

from .attention import AttentionMap, AttentionPooling, TextureEnhancement
from .augmentation import AGDA


class MultiAttentionDetector(nn.Module):
    def __init__(self, backbone, *, warmup_batches=35910, probe=True):
        super().__init__()
        self.backbone = backbone
        self.batch_cnt, self.warmup_batches = 0, warmup_batches
        if probe:
            with torch.no_grad():
                self.features(torch.zeros(1, 3, 256, 256))
        self.AGDA = AGDA()
        self.attention_generation = AttentionMap()
        self.attention_pooling = AttentionPooling()
        self.texture_enhance = TextureEnhancement()
        self.projection_local = nn.Sequential(
            nn.Linear(512, 256), nn.Hardswish(), nn.Linear(256, 256), nn.Hardswish()
        )
        self.projection_final = nn.Sequential(nn.Linear(1792, 256), nn.Hardswish())
        self.ensemble_classifier_fc = nn.Sequential(
            nn.Linear(512, 256), nn.Hardswish(), nn.Linear(256, 2)
        )
        self.dropout = nn.Dropout(0.25, inplace=True)
        self.dropout_final = nn.Dropout(0.5, inplace=True)  # Native unused module.

    def features(self, images):
        trunk = self.backbone.efficientnet
        features = trunk._bn0(trunk._conv_stem(images))
        outputs = {}
        endpoints = {1: "b1", 5: "b2", 9: "b3", 15: "b4", 21: "b5", 29: "b6", 31: "b7"}
        for index, block in enumerate(trunk._blocks):
            features = block(features)
            if index in endpoints:
                outputs[endpoints[index]] = features
        # Native named_children traversal stops before _avg_pooling; neither
        # stem nor head executes _swish, which is registered later in the trunk.
        features = trunk._bn1(trunk._conv_head(features))
        outputs["final"] = features
        outputs["logit"] = self.backbone.last_layer(
            F.adaptive_avg_pool2d(features, 1).view(features.shape[0], -1)
        )
        return outputs

    def forward_outputs(self, images, *, inference=False):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("MultiAttention requires native RGB256 geometry")
        self.batch_cnt += 1  # Deliberately includes inference, matching source.
        if self.batch_cnt <= self.warmup_batches:
            layers = self.features(images)
            return {"cls": layers["logit"], "feat": layers["final"]}
        if not inference:
            with torch.no_grad():
                layers = self.features(images)
                attentions = self.attention_generation(layers["b5"])
                images, _ = self.AGDA.agda(images, attentions)
        layers = self.features(images)
        attentions = self.attention_generation(layers["b5"])
        enhanced, reduced = self.texture_enhance(layers["b2"], attentions)
        local = self.attention_pooling(enhanced, attentions)
        local = self.projection_local(self.dropout(local).view(local.shape[0], -1))
        global_features = self.attention_pooling(
            layers["final"], attentions.sum(dim=1, keepdim=True), norm=1
        ).squeeze(1)
        global_features = F.hardswish(self.projection_final(global_features))
        logits = self.ensemble_classifier_fc(torch.cat((local, global_features), dim=1))
        return {
            "cls": logits,
            "feat": layers["final"],
            "attentions": attentions,
            "feature_maps_d": reduced,
        }

    def forward(self, images, *, inference=False):
        return self.forward_outputs(images, inference=inference)["cls"]
