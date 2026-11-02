"""Explicit bridge from the broken nested source config to disjoint Adam groups."""

import torch

from detectors._shared.trainer import ImageClassificationTrainer, number
from engine.runtime import release_cpu_memory
from .checkpoint import restore_weights


class EffluxletTrainer(ImageClassificationTrainer):
    def load_selected_weights(self, weights):
        restore_weights(self.model, weights)
        release_cpu_memory()

    def compute_loss(self, images, labels):
        return self.model.loss_terms(self.model.forward_outputs(images), labels)[
            "overall"
        ]

    def build_optimizer(self, settings):
        if settings.get("optimizer", "adam") != "adam":
            raise ValueError(
                "Effluxlet configuration bridge supports its two-group Adam recipe"
            )
        groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and not n.startswith("wavelets_branch.")
                ],
                "lr": number(settings, "learning_rate", 5e-6, strict=True),
                "group_name": "all_except_wavelets",
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and n.startswith("wavelets_branch.")
                ],
                "lr": number(settings, "wavelets_learning_rate", 1e-4, strict=True),
                "group_name": "wavelets_branch",
            },
        ]
        return torch.optim.Adam(
            groups,
            weight_decay=number(settings, "weight_decay", 5e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
