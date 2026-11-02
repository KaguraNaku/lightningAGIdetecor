"""LSDA's original B4 stem/1000-way head, plus its fresh512-channel projection."""

from torch import nn

from .efficientnetb4 import B4Trunk


class LSDAB4(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = B4Trunk()
        self.conv = nn.Conv2d(1792, 512, 1)

    def features(self, images):
        return self.conv(self.model.extract_features(images))

    def forward(self, images):
        return self.features(images)


def load_lsda_b4(model_id):
    return LSDAB4()
