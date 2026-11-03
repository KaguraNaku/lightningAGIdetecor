"""Native FTCN clip probability and pooled temporal-feature outputs."""

from detectors.i3d.detector import I3DDetector


class FTCNDetector(I3DDetector):
    def forward_outputs(self, clips):
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (16, 3, 224, 224):
            raise ValueError("The selected FTCN recipe requires NTCHW RGB224/T16 clips")
        probabilities, features = self.resnet(clips.permute(0, 2, 1, 3, 4))
        return {"cls": probabilities, "prob": probabilities, "feat": features}

    def forward(self, clips):
        return self.forward_outputs(clips)["cls"]
