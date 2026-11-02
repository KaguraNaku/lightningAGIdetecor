"""Native UCF common/specific features, ordered reconstructions and objectives."""

import random

import torch
from torch import nn

from .losses import ContrastiveLoss
from .reconstruction import ConditionalUNet


class Head(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(256, 512), nn.LeakyReLU(inplace=True), nn.Linear(512, outputs)
        )

    def forward(self, value):
        features = self.pool(value).view(value.size(0), -1)
        return self.do(self.mlp(features)), features


class Conv2d1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
        )

    def forward(self, value):
        return self.conv2d(value)


class UCFDetector(nn.Module):
    def __init__(self, encoder_f, encoder_c, *, specific_classes=5):
        super().__init__()
        self.encoder_f, self.encoder_c = encoder_f, encoder_c
        self.num_classes, self.encoder_feat_dim, self.half_fingerprint_dim = 2, 512, 256
        # The native objective map is an ordinary dict, not registered modules.
        self.loss_func = {
            "cls": nn.CrossEntropyLoss(),
            "spe": nn.CrossEntropyLoss(),
            "con": ContrastiveLoss(3.0),
            "rec": nn.L1Loss(),
        }
        self.prob, self.label, self.correct, self.total = [], [], 0, 0
        self.lr, self.do, self.pool = (
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.con_gan = ConditionalUNet()
        self.head_spe, self.head_sha = Head(specific_classes), Head(2)
        self.block_spe, self.block_sha = Conv2d1x1(), Conv2d1x1()

    def forward_outputs(self, images, *, inference=False, labels=None):
        forgery = self.encoder_f.features(images)
        content = self.encoder_c.features(images)
        specific, shared = self.block_spe(forgery), self.block_sha(forgery)
        if inference:
            logits, features = self.head_sha(shared)
            self.head_spe(specific)  # Native inference still executes this head.
            probability = logits.softmax(1)[:, 1]
            # Optional native observers are not the unified evaluator's metrics.
            if labels is not None:
                self.prob.append(probability.detach().squeeze().cpu().numpy())
                self.label.append(labels.detach().squeeze().cpu().numpy())
                self.correct += (logits.argmax(1) == (labels >= 1)).sum().item()
                self.total += labels.size(0)
            return {"cls": logits, "prob": probability, "feat": features}
        batch = shared.size(0)
        if random.random() < 0.7:
            indices = list(range(batch // 2))
            random.shuffle(indices)
            shared[: batch // 2] = shared[indices]
            indices = list(range(batch // 2, batch))
            random.shuffle(indices)
            shared[batch // 2 :] = shared[indices]
        all_forgery = torch.cat((specific, shared), dim=1)
        real_forgery, fake_forgery = all_forgery.chunk(2, dim=0)
        real_content, fake_content = content.chunk(2, dim=0)
        fake_self = self.con_gan(fake_forgery, fake_content)
        real_self = self.con_gan(real_forgery, real_content)
        fake_real = self.con_gan(fake_forgery, real_content)
        real_fake = self.con_gan(real_forgery, fake_content)
        specific_logits, specific_features = self.head_spe(specific)
        logits, features = self.head_sha(shared)
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": features,
            "cls_spe": specific_logits,
            "prob_spe": specific_logits.softmax(1)[:, 1],
            "feat_spe": specific_features,
            "feat_content": content,
            "recontruction_imgs": (fake_real, real_fake, fake_self, real_self),
        }

    def loss_terms(self, outputs, images, labels, specific_labels):
        real, fake = images.chunk(2, dim=0)
        fake_real, real_fake, fake_self, real_self = outputs["recontruction_imgs"]
        common = self.loss_func["cls"](outputs["cls"], labels)
        specific = self.loss_func["spe"](outputs["cls_spe"], specific_labels)
        rec = self.loss_func["rec"]
        reconstruction = (
            rec(fake, fake_self)
            + rec(real, real_self)
            + rec(fake, real_fake)
            + rec(real, fake_real)
        )
        contrastive = self.loss_func["con"](
            outputs["feat"], outputs["feat_spe"], specific_labels
        )
        return {
            "overall": common
            + 0.1 * specific
            + 0.3 * reconstruction
            + 0.05 * contrastive,
            "common": common,
            "specific": specific,
            "reconstruction": reconstruction,
            "contrastive": contrastive,
        }

    def forward(self, images):
        return self.forward_outputs(images, inference=True)["cls"]
