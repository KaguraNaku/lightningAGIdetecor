"""Literal channel-wise native LSDA latent augmentations, not comment corrections."""

import random

import torch


def augment_domains(groups):
    domain_count = len(groups[0])
    means, hardest = [], []
    for index in range(domain_count):
        samples = torch.cat([group[index] for group in groups], dim=0)
        means.append(torch.mean(samples, dim=0))
    for index in range(domain_count):
        samples = torch.cat([group[index] for group in groups], dim=0)
        distances = torch.tensor(
            [torch.norm(value - means[index]) for value in samples]
        )
        hardest.append(samples[torch.argmax(distances)])
    result = []
    for group in groups:
        domains = []
        for index, channels in enumerate(group):
            # Native random.choice selects ONE rule per domain, then that rule
            # draws its scalar(s) separately for each spatial channel map.
            choice = random.choice(range(4))
            augmented = []
            for value in channels:
                if choice == 0:
                    value = value + random.random() * (hardest[index] - value)
                elif choice == 1:
                    value = value + random.random() * (value - means[index])
                elif choice == 2:
                    sigma, strength = random.random(), random.random()
                    noise = torch.randn_like(value) * sigma
                    value = value + strength * noise
                else:
                    value = value + random.random() * (channels[0] - channels[1])
                augmented.append(value)
            domains.append(torch.stack(augmented))
        result.append(torch.stack(domains))
    return torch.stack(result)


def mixup_in_latent_space(data):
    batch, domain_count = data.shape[:2]
    result = torch.zeros_like(data)
    for index in range(batch):
        order = torch.randperm(domain_count)
        alpha = torch.rand(1) * 1.5 + 0.5
        weight = torch.distributions.beta.Beta(alpha, alpha).sample().to(data.device)
        result[index] = weight * data[index] + (1 - weight) * data[index, order]
    return result


def cosine_similarity_loss(first, second, dim=1, eps=1e-8):
    first = first / (first.norm(dim=dim, keepdim=True) + eps)
    second = second / (second.norm(dim=dim, keepdim=True) + eps)
    return 1 - (first * second).sum(dim=dim)
