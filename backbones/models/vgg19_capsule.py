"""Independent VGG19 prefix with the native pretrained constructor RNG order."""

from torch import nn


class _VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for value in (
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ):
            if value == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.extend((nn.Conv2d(channels, value, 3, padding=1), nn.ReLU(True)))
                channels = value
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
        )
        # torchvision's pretrained=True forces init_weights=False. The default
        # Conv/Linear resets above still consume RNG, including discarded layers.


class VGG19Features(nn.Module):
    def __init__(self, *, weights=None):
        super().__init__()
        full = _VGG19()
        if weights is not None:
            full.load_state_dict(weights, strict=True)
        self.vgg_1 = nn.Sequential(*list(full.features.children())[:19])
        self.vgg_1.eval()
        # The source does not freeze VGG parameters. Parent train() later
        # recursively changes this initial evaluation flag as usual.

    def forward(self, images):
        if tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("CapsuleNet requires native RGB256 images")
        return self.vgg_1(images)


def load_vgg19_capsule(model_id, **kwargs):
    return VGG19Features(**kwargs)
