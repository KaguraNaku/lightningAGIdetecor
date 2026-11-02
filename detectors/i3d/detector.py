"""I3D NTCHW clip classification with native probability BCE semantics."""

from torch import nn


class I3DDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.resnet = backbone
        self.loss_func = nn.BCELoss()

    def forward(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (16, 3, 224, 224):
            raise ValueError("The selected I3D recipe requires NTCHW RGB224/T16 clips")
        return self.resnet(clips.permute(0, 2, 1, 3, 4))

    def forward_outputs(self, clips):
        probability = self(clips)
        return {"cls": probability, "prob": probability, "feat": probability}

    def loss_terms(self, outputs, labels):
        if labels.shape != (outputs["cls"].shape[0],):
            raise ValueError("I3D needs one binary label per clip")
        if not labels.is_meta and not ((labels == 0) | (labels == 1)).all():
            raise ValueError("I3D labels must be binary")
        return {"overall": self.loss_func(outputs["cls"].view(-1), labels.float())}
