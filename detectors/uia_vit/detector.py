"""Native labeled forward, covariance cadence and observable patch-bank state."""

import torch
from torch import nn

from .statistics import UIAStatistics


class UIAViTDetector(nn.Module):
    def __init__(self, model, *, batch_per_epoch=1796, epochs=10, bank_directory=None):
        super().__init__()
        if (
            type(batch_per_epoch) is not int
            or batch_per_epoch < 2
            or type(epochs) is not int
            or epochs < 1
        ):
            raise ValueError("UIA requires batch_per_epoch >= 2 and epochs >= 1")
        self.model = model
        self.batch_per_epoch, self.num_epoch = batch_per_epoch, epochs
        self.batch_cnt = 0
        self.statistics = UIAStatistics(bank_directory)

    def forward_outputs(self, images, labels):
        if (
            labels.shape != (images.shape[0],)
            or not ((labels == 0) | (labels == 1)).all()
        ):
            raise ValueError("UIA labeled forward requires aligned binary labels")
        if (
            self.model.training
            and self.batch_cnt != 0
            and self.batch_cnt % (self.batch_per_epoch // 2) == 0
        ):
            self.statistics.fit()
        step = (
            self.batch_cnt / (self.batch_per_epoch * self.num_epoch)
            if self.model.training
            else 1
        )
        logits, patches, raw = self.model(images, step=step)
        self.statistics.collect(patches, labels)
        real = torch.where(labels == 0.0)[0]
        fake = torch.where(labels == 1.0)[0]
        real_attention = torch.sigmoid(torch.mean(raw[real, :, 1:, 1:], dim=1))
        fake_attention = torch.sigmoid(torch.mean(raw[fake, :, 1:, 1:], dim=1))
        self.batch_cnt += 1
        return {
            "cls": logits,
            "feat": patches,
            "attention_map_real": real_attention,
            "attention_map_fake": fake_attention,
        }

    def score_only(self, images):
        # Label-free deployment has no reference bank mutation to reproduce.
        # The unified labeled evaluator uses forward_outputs instead.
        if self.model.training:
            raise ValueError("UIA label-free scoring requires evaluation mode")
        return self.model(images, step=1)[0]

    def forward(self, images, labels=None):
        if labels is None:
            return self.score_only(images)
        return self.forward_outputs(images, labels)["cls"]

    def method_state_dict(self):
        return {"batch_cnt": self.batch_cnt, **self.statistics.state_dict()}

    def load_method_state_dict(self, state):
        state = dict(state)
        count = state.pop("batch_cnt")
        if type(count) is not int or count < 0:
            raise ValueError("UIA batch count must be a nonnegative integer")
        self.statistics.load_state_dict(state)
        self.batch_cnt = count
