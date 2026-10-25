import torch
from torch import nn
class FeatureMapping(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers_count = config["layers_count"]
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]

        self.config = config
        self.register_buffer(
            "means", torch.zeros(layers_count, input_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "scales", torch.ones(layers_count, input_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "basis", torch.zeros(layers_count, input_dim, output_dim, dtype=torch.float32)
        )

        self.layers_count = layers_count
        self.input_dim = input_dim
        self.output_dim = output_dim

    @torch.no_grad()
    def fit(self, features, labels):
        eps = self.config["eps"]

        for layer in range(self.layers_count):
            layer_feature = features[:, layer, :]
            real_features = layer_feature[labels == 0]
            real_counts = len(real_features)

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

            basis = eigenvectors[:, -self.output_dim:].flip(1).contiguous()

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
        outputs = []
        for layer in range(self.layers_count):
            layer_features = features[:, layer, :]
            standardized = (layer_features - self.means[layer]) / self.scales[layer]
            projection = standardized @ self.basis[layer]
            outputs.append(projection)
        return torch.cat(outputs, dim=1)