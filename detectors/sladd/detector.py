"""Native main Xception plus policy synthesis and pair-switch/flip assembly."""

import torch
from torch import nn

from .losses import detector_losses


class SLADDDetector(nn.Module):
    def __init__(self, backbone, synthesizer):
        super().__init__()
        self.backbone = backbone
        self.synthesizer = synthesizer

    def detector_parameters(self):
        # Keep ordinary nn.Module.parameters() complete for state inspection;
        # select the exact native filtered order for the main optimizer.
        for name, parameter in self.named_parameters():
            if "synthesizer" not in name:
                yield parameter

    def prepare_pairs(self, images, landmarks, masks, labels):
        device = self.backbone.conv1.weight.device
        images, landmarks, masks, labels = (
            value.to(device) for value in (images, landmarks, masks, labels)
        )
        half = len(images) // 2
        real, fake = images[:half], images[half:]
        real_landmarks, fake_landmarks = landmarks[:half], landmarks[half:]
        real_masks, fake_masks = masks[:half], masks[half:]
        real_labels, fake_labels = labels[:half], labels[half:]
        # Native CPU draw and advanced-index tuple assignment intentionally
        # mutate the collated batch views before the policy is evaluated.
        switch = torch.randint(0, 2, (real.shape[0],)).bool()
        (
            real[switch],
            fake[switch],
            real_landmarks[switch],
            fake_landmarks[switch],
            real_masks[switch],
            fake_masks[switch],
            real_labels[switch],
            fake_labels[switch],
        ) = (
            fake[switch],
            real[switch],
            fake_landmarks[switch],
            real_landmarks[switch],
            fake_masks[switch],
            real_masks[switch],
            fake_labels[switch],
            real_labels[switch],
        )
        (
            log_prob,
            entropy,
            generated,
            target_mask,
            generated_labels,
            type_labels,
            magnitude_labels,
            magnitude_masks,
        ) = self.synthesizer(
            real,
            fake,
            real_landmarks,
            fake_landmarks,
            real_masks,
            fake_masks,
            label=labels[:half],
        )
        (
            generated,
            target_mask,
            generated_labels,
            type_labels,
            magnitude_labels,
            magnitude_masks,
        ) = (
            value.to(device)
            for value in (
                generated,
                target_mask,
                generated_labels,
                type_labels,
                magnitude_labels,
                magnitude_masks,
            )
        )
        flipped = torch.flip(generated, (3,)).detach().clone()
        flipped_mask = torch.flip(target_mask, (3,)).detach().clone()
        return {
            "images": torch.cat((generated, flipped)),
            "alt_mask": torch.cat((target_mask, flipped_mask)),
            "label": torch.cat((generated_labels, generated_labels)),
            "type_label": torch.cat((type_labels, type_labels)),
            "mag_label": torch.cat((magnitude_labels, magnitude_labels)),
            "mag_mask": torch.cat((magnitude_masks, magnitude_masks)),
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def forward_outputs(
        self, images, *, labels=None, landmarks=None, masks=None, inference=False
    ):
        if inference:
            generated = images
            outputs = {} if labels is None else {"label": labels}
        else:
            if labels is None or landmarks is None or masks is None:
                raise ValueError(
                    "SLADD training requires paired labels, landmarks and HWC masks"
                )
            outputs = self.prepare_pairs(images, landmarks, masks, labels)
            generated = outputs.pop("images")
        features, spatial = self.backbone.features(generated)
        logits, pooled = self.backbone.classifier(features)
        if not inference:
            outputs.update(
                map=self.backbone.map(spatial),
                type=self.backbone.type_fc(pooled),
                mag=self.backbone.mag_fc(pooled),
            )
        return {
            "cls": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "feat": features,
            **outputs,
        }

    def forward(self, images):
        return self.forward_outputs(images, inference=True)["cls"]

    @staticmethod
    def loss_terms(outputs, labels=None):
        # Generated labels in outputs are authoritative, as in the source.
        return detector_losses(outputs)
