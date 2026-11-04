"""Literal native categorical/continuous-magnitude policy sampling arithmetic."""

import torch
from torch.nn import functional as F


def policy_action(logits):
    if logits.shape[1] != 1:
        probabilities = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(log_prob * probabilities).sum(1, keepdim=False)
        actions = probabilities.multinomial(num_samples=1).detach()
        selected = log_prob.gather(1, actions)
    else:
        probabilities = torch.sigmoid(logits)
        # Preserve the second sigmoid and log(sigmoid),not log-sigmoid or a
        # Bernoulli draw. Native magnitude entropy is only -p*log(p).
        log_prob = torch.log(torch.sigmoid(logits))
        entropy = -(log_prob * probabilities).sum(1, keepdim=False)
        actions, selected = probabilities, log_prob
    return entropy, selected[:, 0], actions[:, 0]
