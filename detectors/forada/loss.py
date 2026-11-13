"""Author's full objective; never silently replace it with CE-only training."""
import torch
from torch import nn
from torch.nn import functional as F

class ForAdaLoss(nn.Module):
    def forward(self, outputs, data):
        required = {"label", "xray", "patch_label", "clip_patch_label", "if_boundary"}
        if not required <= data.keys():
            raise ValueError("Native ForAda training requires mask-derived auxiliary targets")
        if not (data["label"] == 1).any():
            raise ValueError("Native ForAda contrastive objective is undefined for real-only batches")
        losses = {
            "cls": F.cross_entropy(outputs["cls"].float(), data["label"]),
            "xray": F.mse_loss(outputs["xray_pred"].squeeze().float(), data["xray"].squeeze().float()),
            "intra": outputs["loss_intra"], "loss_clip": outputs["loss_clip"],
        }
        losses["overall"] = 10 * losses["cls"] + 200 * losses["xray"] + 20 * losses["intra"] + 10 * losses["loss_clip"]
        if not torch.isfinite(losses["overall"]):
            raise ValueError("Nonfinite native ForAda objective")
        return losses
