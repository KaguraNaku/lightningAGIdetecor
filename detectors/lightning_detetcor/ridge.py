import torch


def solve_ridge(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    *,
    width: int = 64,
    ridge_lambda: float = 1.0,
):
    """Solve a ridge system with layer-specific penalties.

    Parameters
    ----------
    gram : torch.Tensor of shape (n_features, n_features)
        Feature Gram matrix.
    rhs : torch.Tensor of shape (n_features,)
        Feature-target cross-product with centered targets.
    width : int, default=64
        Number of features per layer.
    ridge_lambda : float, default=1.0
        Positive multiplier for the layer-specific penalties.

    Returns
    -------
    weights : torch.Tensor of shape (n_features,)
        Fitted regression coefficients.
    """
    if width < 1 or gram.shape[0] % width:
        raise ValueError("width must be positive and divide n_features")

    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")

    diagonal = torch.diag(gram)

    # group by layers
    blocks = diagonal.reshape(-1, width)
    layer_penalties = blocks.sum(dim=1) / width
    penalty = ridge_lambda * layer_penalties.repeat_interleave(width)

    if not torch.isfinite(penalty).all():
        raise ValueError("Regularization penalties must be finite")

    if not (penalty > 0).all():
        raise ValueError("Degenerate feature block")

    system_matrix = gram + torch.diag(penalty)
    return torch.linalg.solve(system_matrix, rhs)


@torch.no_grad()
def fit_ridge(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    width: int = 64,
    ridge_lambda: float = 1.0,
):
    """Fit ridge regression from features and labels.

    Parameters
    ----------
    features : torch.Tensor of shape (n_samples, n_features)
        Floating-point features, concatenated in layer order.
    labels : torch.Tensor of shape (n_samples,)
        Floating-point targets with the same dtype and device as features.
    width : int, default=64
        Number of features per layer.
    ridge_lambda : float, default=1.0
        Positive multiplier for the layer-specific penalties.

    Returns
    -------
    weights : torch.Tensor of shape (n_features,)
        Fitted regression coefficients.
    """
    gram = features.T @ features
    targets = labels - labels.mean()
    rhs = features.T @ targets

    return solve_ridge(
        gram,
        rhs,
        width=width,
        ridge_lambda=ridge_lambda,
    )