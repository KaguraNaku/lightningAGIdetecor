"""Native post-forward21-batch spatial/temporal freezing state machine."""

from torch import nn

from detectors.i3d.detector import I3DDetector


class AltFreezingDetector(I3DDetector):
    def __init__(self, backbone):
        super().__init__(backbone)
        self.conv_dict = {"temporal": [], "spatial": []}
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv3d):
                if module.kernel_size == (3, 1, 1):
                    self.conv_dict["temporal"].append(name)
                if module.kernel_size == (1, 3, 3):
                    self.conv_dict["spatial"].append(name)
        self.train_batch_cnt = 0

    def alternate_mode(self, target_mode):
        if target_mode not in ("temporal", "spatial"):
            raise ValueError("AltFreezing target mode must be temporal or spatial")
        modules = dict(self.resnet.named_modules())
        for mode in ("temporal", "spatial"):
            for name in self.conv_dict[mode]:
                layer = modules[name]
                layer.weight.requires_grad_(target_mode == mode)
                if layer.bias is not None:
                    layer.bias.requires_grad_(target_mode == mode)

    def forward(self, clips, *, inference=False):
        probability = super().forward(clips)
        # Preserve placement AFTER the differentiable forward. Moving this
        # before the backbone changes which weights occur in the autograd graph.
        # The native inference argument is independent of nn.Module.training.
        if not inference:
            if self.train_batch_cnt % 21 == 0:
                self.alternate_mode("spatial")
            elif self.train_batch_cnt % 21 == 1:
                self.alternate_mode("temporal")
            self.train_batch_cnt += 1
        return probability

    def forward_outputs(self, clips, *, inference=False):
        probability = self(clips, inference=inference)
        return {"cls": probability, "prob": probability, "feat": probability}

    def schedule_state_dict(self):
        return {
            "version": 1,
            "train_batch_cnt": self.train_batch_cnt,
            "requires_grad": {
                name: parameter.requires_grad
                for name, parameter in self.named_parameters()
            },
        }

    def load_schedule_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported AltFreezing schedule state")
        count = state.get("train_batch_cnt")
        flags = state.get("requires_grad")
        parameters = dict(self.named_parameters())
        if (
            type(count) is not int
            or count < 0
            or not isinstance(flags, dict)
            or set(flags) != set(parameters)
            or any(type(flag) is not bool for flag in flags.values())
        ):
            raise ValueError("AltFreezing needs a complete counter/gradient-flag state")
        self.train_batch_cnt = count
        for name, parameter in parameters.items():
            parameter.requires_grad_(flags[name])
