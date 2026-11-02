"""Native HRNet boundary regression followed by mask-based classification."""

from torch import nn
from torch.nn import functional as F


class FaceXrayDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.post_process = nn.Sequential(
            nn.Conv2d(720, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward_outputs(self, images):
        features = self.backbone.features(images)
        mask = self.post_process(features)
        pooled = F.adaptive_avg_pool2d(mask, 128).view(mask.size(0), -1)
        logits = self.fc(pooled)
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": pooled,
            "mask_pred": mask,
        }

    def forward(self, images):
        return self.forward_outputs(images)["cls"]

    def loss_terms(self, outputs, labels, *, mask=None):
        classification = F.cross_entropy(outputs["cls"], labels)
        if mask is None:
            return {"overall": classification, "cls": classification}
        boundary = F.mse_loss(
            outputs["mask_pred"].squeeze().float(),
            mask.to(outputs["mask_pred"].device).squeeze().float(),
        )
        return {
            "overall": classification + 1000.0 * boundary,
            "mask": boundary,
            "cls": classification,
        }
