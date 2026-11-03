"""STIL clip consensus and two-logit CE, composed over a supplied SCNet."""

import torch
from torch import nn


class STILClipModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.num_class = 2
        self.num_segment = 8
        self.add_softmax = False
        if backbone.num_segments != self.num_segment:
            raise ValueError("STIL requires the native eight temporal segments")
        self.base_model = backbone
        # The original builds/initializes its1000-way backbone,loads pretraining,
        # then initializes this replacement. Task must preserve that load order.
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, self.num_class)

    def forward(self, packed):
        logits = self.base_model(packed.view((-1, 3) + packed.size()[2:]))
        return logits.view(-1, self.num_segment, self.num_class).mean(1, keepdim=False)


class STILDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = STILClipModel(backbone)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (8, 3, 224, 224):
            raise ValueError("STIL requires NTCHW RGB224/T8 clips")
        batch, time, channels, height, width = clips.shape
        return self.model(clips.view(batch, time * channels, height, width))

    def forward_outputs(self, clips):
        logits = self(clips)
        probability = logits.softmax(dim=1)[:, 1]
        # Native STIL names probability as feat too; retain that alias.
        return {"cls": logits, "prob": probability, "feat": probability}

    def loss_terms(self, outputs, labels):
        if labels.dtype != torch.int64 or labels.shape != (len(outputs["cls"]),):
            raise ValueError("STIL requires one int64 binary label per clip")
        if not labels.is_meta and not ((labels == 0) | (labels == 1)).all():
            raise ValueError("STIL labels must be binary")
        return {"overall": self.loss_func(outputs["cls"], labels.long())}
