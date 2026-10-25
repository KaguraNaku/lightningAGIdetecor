import torch

from .ridge import fit_ridge


def search_layers(width: int = 64, topk: int = 8, **ridge_args):
    weights = fit_ridge(width=width, **ridge_args)

    scores = weights.reshape(-1, width).norm(dim=1)
    if not isinstance(topk, int) or not 1 <= topk <= len(scores):
        raise ValueError("topk must be between 1 and the number of layers")
    layers = torch.argsort(scores, descending=True)[:topk]

    return sorted(layers.tolist()), scores
