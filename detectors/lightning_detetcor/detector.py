from backbones.load_backbone import load_backbone
from mapping import FeatureMapping
class LightningDetector():
    def __init__(self, config):
        self.backbone = load_backbone("openai/clip-vit-large-patch14", "vision")
        mapping_config = {
            'layers_count': config['layers_count'],
            'input_dim': config['input_dim'],
            'output_dim': config['output_dim'],
            'eps': config['eps'],
        }
        self.mapping = FeatureMapping(mapping_config)
        self.classifier = 

    def search_layers(self):
        gram = z.T @ z
        penalty = (gram.diagonal().reshape(-1, width).sum(1) / width).repeat_interleave(
            width
        )
        if not (penalty > 0).all():
            raise ValueError("Degenerate feature block")
        gram.diagonal().add_(penalty)
        weights = torch.linalg.solve(gram, z.T @ (y - y.mean()))
        scores = weights.reshape(-1, width).norm(dim=1)
        return sorted(torch.argsort(scores, descending=True)[:topk].tolist()), scores

LightningDetector()
print()