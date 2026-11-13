"""Compose frozen CLIP, the query adapter, reconstructed attention and heads."""
from torch import nn
from torch.nn import functional as F
from .adapter import Adapter
from .attention import RecAttnClip
from .layers import MaskPostXrayProcess, PostClipProcess

class ForAdaDetector(nn.Module):
    def __init__(self, backbone, *, queries=128, device="cpu"):
        super().__init__()
        self.clip_model = backbone
        width = backbone.visual.width
        self.adapter = Adapter(
            vit_name="vit_tiny_patch16_224", num_quires=queries,
            fusion_map={1: 1, 2: 8, 3: 15}, mlp_dim=256, mlp_out_dim=128,
            head_num=width // 64, device=device,
        )
        self.rec_attn_clip = RecAttnClip(backbone.visual, queries, device=device)
        self.masked_xray_post_process = MaskPostXrayProcess(queries)
        self.clip_post_process = PostClipProcess(queries, embed_dim=768)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

    def forward_outputs(self, data, *, inference):
        images = data["image"]
        if images.shape[1:] != (3, 256, 256):
            raise ValueError("ForAda requires RGB256 tensors")
        # Author modules hold an explicit device attribute; synchronize after .to().
        self.adapter.device = self.rec_attn_clip.device = images.device
        clip_images = F.interpolate(images, (224, 224), mode="bilinear", align_corners=False)
        features = self.clip_model.extract_features(clip_images, self.adapter.fusion_map.values())
        biases, xrays, loss_intra = self.adapter(data, features, inference)
        queries, loss_clip = self.rec_attn_clip(data, features, biases[-1], inference, normalize=True)
        boundaries = data["if_boundary"].to(images.device)
        xray = self.masked_xray_post_process(xrays[-1], boundaries)
        # The native squeeze drops N=1; the migrated head only squeezes the channel axis.
        logits = self.clip_post_process(queries.float())
        return {"cls": logits, "xray_pred": xray, "loss_intra": loss_intra, "loss_clip": loss_clip}

    def forward(self, images):
        # Ground-truth masks/labels are never used to compute classification logits.
        return self.forward_outputs({
            "image": images,
            "if_boundary": images.new_ones((len(images), 256)),
        }, inference=True)["cls"]
