"""Task→Detector→Backbone scoring with the historically validated readouts."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel
from .analytic import LAYERS


class DualReadoutBackbone(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        hidden = self.vision(pixel_values=images, output_hidden_states=True).hidden_states[1:]
        cls = torch.stack([F.normalize(hidden[int(l)][:, 0], dim=-1) for l in LAYERS], 1)
        mean = torch.stack([F.normalize(hidden[int(l)][:, 1:].mean(1), dim=-1) for l in LAYERS], 1)
        return {'cls': cls, 'mean': mean}


class Fixed8DualDetector(nn.Module):
    def __init__(self, backbone, coefficients):
        super().__init__()
        self.backbone = backbone
        self.names = list(coefficients)
        for branch in ['cls', 'mean']:
            value = torch.stack([torch.as_tensor(coefficients[n][branch], dtype=torch.float64)
                                 for n in self.names], -1)
            self.register_buffer(branch, value.reshape(-1, len(self.names)))
        self.register_buffer('bias', torch.tensor([float(coefficients[n]['bias']) for n in self.names], dtype=torch.float64))

    @torch.inference_mode()
    def forward(self, images):
        features = self.backbone(images)
        result = self.bias.expand(len(images), -1).clone()
        for kind, value in features.items():
            result += value.double().flatten(1) @ getattr(self, kind)
        return result


class Fixed8LightningTask:
    def __init__(self, processor, vision, coefficients):
        self.processor = CLIPImageProcessor.from_dict(processor)
        self.detector = Fixed8DualDetector(DualReadoutBackbone(vision), coefficients)

    @classmethod
    def from_cached(cls, processor, coefficients):
        vision = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14',
            revision='32bd64288804d66eefd0ccbe215aa642df71cc41', local_files_only=True)
        return cls(processor, vision, coefficients)
