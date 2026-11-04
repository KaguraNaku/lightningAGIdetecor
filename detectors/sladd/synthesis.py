"""Native policy network plus detached action-conditioned pixel synthesis.

The renderer can be audited at its real action boundary without substituting
a smaller policy CNN. The complete module always uses the full native encoder.
"""

import cv2
import numpy as np
import torch
from torch import nn

from .encoder import SynthesisTransfer
from .generation import blend_fake_to_real, parse_region
from .sampling import policy_action


def normalized_uint8(image):
    array = np.array(image, dtype=np.uint8)
    value = (
        torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        .float()
        .div(255)
    )
    return value.sub_(0.5).div_(0.5)


def render_actions(
    images,
    fakes,
    real_landmarks,
    fake_landmarks,
    real_masks,
    fake_masks,
    regions,
    kinds,
    magnitude,
    *,
    labels=None,
):
    generated = torch.ones(images.shape)
    masks = np.zeros((images.shape[0], 16, 16))
    if labels is None:
        labels = np.zeros(images.shape[0])
    generated_labels, type_labels, magnitude_masks = [], [], []
    for index in range(images.shape[0]):
        real = np.transpose(images[index].cpu().numpy(), (1, 2, 0)).copy()
        fake = np.transpose(fakes[index].cpu().numpy(), (1, 2, 0)).copy()
        if labels[index] == 0 and kinds[index] != 3:
            mask = parse_region(
                fake,
                regions[index],
                fake_landmarks[index].cpu().numpy(),
                fake_masks[index].cpu().numpy(),
            )
            image, mask = blend_fake_to_real(
                real,
                real_landmarks[index].cpu().numpy(),
                fake,
                mask,
                kinds[index],
                magnitude[index].detach().cpu().numpy(),
            )
            generated[index] = normalized_uint8(image)
            generated_labels.append(1)
            type_labels.append(int(kinds[index].cpu().numpy()))
            magnitude_masks.append(1 if kinds[index] == 2 else 0)
        else:
            generated[index] = normalized_uint8((real + 1) / 2 * 255)
            # Source assumes a three-channel HWC mask at this boundary. Keep
            # the exact indexing and its grayscale-layout failure visible.
            mask = real_masks[index].squeeze(2)[:, :, 0].cpu().numpy()
            generated_labels.append(int(labels[index]))
            type_labels.append(3 if labels[index] == 0 else 4)
            magnitude_masks.append(0)
        if mask is None:
            mask = np.zeros((16, 16))
        masks[index] = cv2.resize(mask, (16, 16), interpolation=cv2.INTER_CUBIC)
    masks = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
    return (
        generated.detach(),
        masks.detach(),
        torch.tensor(generated_labels).detach(),
        torch.tensor(type_labels).detach(),
        magnitude.detach(),
        torch.tensor(magnitude_masks).detach(),
    )


class SLADDSynthesizer(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.netG = SynthesisTransfer(pretrained)

    @staticmethod
    def parse(image, region, landmarks, fake_mask):
        return parse_region(image, region, landmarks, fake_mask)

    def forward(
        self,
        images,
        fakes,
        real_landmarks,
        fake_landmarks,
        real_masks,
        fake_masks,
        *,
        label=None,
    ):
        region_logits, type_logits, magnitude_logits = self.netG(
            torch.cat((images, fakes), 1)
        )
        region_entropy, region_log, regions = policy_action(region_logits)
        type_entropy, type_log, kinds = policy_action(type_logits)
        magnitude_entropy, magnitude_log, magnitude = policy_action(magnitude_logits)
        entropy = region_entropy + type_entropy + magnitude_entropy
        log_prob = region_log + type_log + magnitude_log
        rendered = render_actions(
            images,
            fakes,
            real_landmarks,
            fake_landmarks,
            real_masks,
            fake_masks,
            regions,
            kinds,
            magnitude,
            labels=label,
        )
        return log_prob, entropy, *rendered
