"""Ten statistical capsules and native two-iteration dynamic routing."""

import torch
from torch import nn
from torch.nn import functional as F


def squash(values, dim):
    squared = (values**2).sum(dim=dim, keepdim=True)
    return squared / (1 + squared) * values / torch.sqrt(squared)


class StatsNet(nn.Module):
    def forward(self, values):
        values = values.view(values.shape[0], values.shape[1], -1)
        return torch.stack((values.mean(2), values.std(2)), dim=1)


class View(nn.Module):
    def forward(self, values):
        return values.view(-1, 8)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.capsules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    StatsNet(),
                    nn.Conv1d(2, 8, 5, stride=2, padding=2),
                    nn.BatchNorm1d(8),
                    nn.Conv1d(8, 1, 3, padding=1),
                    nn.BatchNorm1d(1),
                    View(),
                )
                for _ in range(10)
            ]
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def forward(self, features):
        return squash(
            torch.stack([branch(features) for branch in self.capsules], -1), -1
        )


class RoutingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.route_weights = nn.Parameter(torch.randn(2, 10, 4, 8))

    def forward(self, features):
        features = features.transpose(2, 1)
        priors = self.route_weights[:, None] @ features[None, :, :, :, None]
        priors = priors.transpose(1, 0)
        logits = torch.Tensor(torch.zeros(*priors.size())).to(priors.device)
        for iteration in range(2):
            probabilities = F.softmax(logits, dim=2)
            outputs = squash((probabilities * priors).sum(dim=2, keepdim=True), 3)
            if iteration != 1:
                logits = logits + priors * outputs
        outputs = outputs.squeeze()
        if outputs.ndim == 3:
            return outputs.transpose(2, 1).contiguous()
        return outputs.unsqueeze_(0).transpose(2, 1).contiguous()


class CapsuleNetDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.vgg_ext = backbone
        self.fea_ext = FeatureExtractor()
        self.routing_stats = RoutingLayer()

    def forward_outputs(self, images):
        features = self.fea_ext(self.vgg_ext(images))
        classes = F.softmax(self.routing_stats(features), dim=-1)
        prediction = classes.detach().mean(dim=1)
        return {
            "cls": prediction,
            "prob": prediction.softmax(1)[:, 1],
            "feat": features,
            "classes": classes,
        }

    def forward(self, images):
        # Native cls is a detached probability mean, not classification logits.
        return self.forward_outputs(images)["cls"]
