import torch
from torch import nn


class FeatureMapping(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers_count = config["layers_count"]
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]

        if not isinstance(layers_count, int) or layers_count < 1:
            raise ValueError("layers_count must be a positive integer")
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(output_dim, int) or not 1 <= output_dim <= input_dim:
            raise ValueError("output_dim must be between 1 and input_dim")
        if not 0 < config["eps"] < float("inf"):
            raise ValueError("eps must be finite and positive")

        self.config = config
        self.register_buffer(
            "means", torch.zeros(layers_count, input_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "scales", torch.ones(layers_count, input_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "basis",
            torch.zeros(layers_count, input_dim, output_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "selected_layers", torch.arange(layers_count, dtype=torch.int64)
        )

        self.layers_count = layers_count
        self.input_dim = input_dim
        self.output_dim = output_dim

    @torch.no_grad()
    def select_layers(self, layers):
        """Save selected layer identities; retain their original ascending order."""
        layers = torch.as_tensor(layers, device=self.selected_layers.device)
        if layers.ndim != 1 or not len(layers):
            raise ValueError("layers must be a nonempty 1D sequence")
        if layers.dtype not in (torch.int32, torch.int64):
            raise ValueError("layer indices must be integers")
        if not ((layers >= 0) & (layers < self.layers_count)).all():
            raise ValueError("layer indices are out of range")
        if len(layers.unique()) != len(layers):
            raise ValueError("layer indices must be unique")
        self.selected_layers = layers.long().sort().values.clone()
        return self

    def _prepare_features(self, features):
        expected = (self.layers_count, self.input_dim)
        if features.ndim != 3 or features.shape[1:] != expected or not len(features):
            raise ValueError(
                f"features must have shape (N, {expected[0]}, {expected[1]})"
            )
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        return features.to(self.means)

    @torch.no_grad()
    def fit(self, features, labels):
        features = self._prepare_features(features)
        labels = labels.to(device=features.device)
        if labels.shape != (len(features),):
            raise ValueError("labels must contain one value per sample")
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("labels must contain only raw 0/1 targets")
        if not (labels == 0).any():
            raise ValueError("fitting requires at least one real sample")
        eps = self.config["eps"]

        for layer in range(self.layers_count):
            layer_feature = features[:, layer, :]
            real_features = layer_feature[labels == 0]

            # mean = real_features.sum(dim=0) / (real_counts + eps)
            # variance = real_features.square().sum(dim=0) / (real_counts + eps) - mean.square()
            # scale = (variance + eps).sqrt()

            # NOTE: real_features must contain at least one sample.
            mean = real_features.mean(dim=0)
            variance = (real_features - mean).square().mean(dim=0)
            scale = (variance + eps).sqrt()

            standardized = (layer_feature - mean) / scale

            gram = standardized.T @ standardized
            # Rayleigh-Ritz theory
            _, eigenvectors = torch.linalg.eigh(gram)

            basis = eigenvectors[:, -self.output_dim :].flip(1).contiguous()

            # Make the largest-magnitude entry in each basis vector positive.
            indices = basis.abs().argmax(dim=0)
            columns = torch.arange(self.output_dim, device=basis.device)

            # TODO: need to explain why
            basis *= basis[indices, columns].sign()

            self.means[layer].copy_(mean)
            self.scales[layer].copy_(scale)
            self.basis[layer].copy_(basis)

        return self

    def forward(self, features):
        features = self._prepare_features(features)
        outputs = []
        for layer in self.selected_layers.tolist():
            layer_features = features[:, layer, :]
            standardized = (layer_features - self.means[layer]) / self.scales[layer]
            projection = standardized @ self.basis[layer]
            outputs.append(projection)
        return torch.cat(outputs, dim=1)
