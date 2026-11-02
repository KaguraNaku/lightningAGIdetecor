"""Independent single-pathway I3D ResNet-50 for the native RGB224/T16 recipe.

Keep native state names and constructor/global initialization order. This module
does not import SlowFast, fvcore, the old checkout or other detector backends.
The fixed-pool head returns probabilities in both train and evaluation modes.
"""

import torch
from torch import nn


class I3DStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(
            3, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False
        )
        self.bn = nn.BatchNorm3d(64, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pool_layer = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))

    def forward(self, value):
        return self.pool_layer(self.relu(self.bn(self.conv(value))))


class SinglePathStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.pathway0_stem = I3DStem()

    def forward(self, value):
        return self.pathway0_stem(value)


class I3DBottleneck(nn.Module):
    def __init__(self, inputs, outputs, inner, temporal_kernel, stride):
        super().__init__()
        self.a = nn.Conv3d(
            inputs,
            inner,
            (temporal_kernel, 1, 1),
            stride=(1, 1, 1),
            padding=(temporal_kernel // 2, 0, 0),
            bias=False,
        )
        self.a_bn = nn.BatchNorm3d(inner, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            inner,
            inner,
            (1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            groups=1,
            bias=False,
            dilation=(1, 1, 1),
        )
        self.b_bn = nn.BatchNorm3d(inner, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        self.c = nn.Conv3d(
            inner,
            outputs,
            (1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.c_bn = nn.BatchNorm3d(outputs, eps=1e-5, momentum=0.1)
        self.c_bn.transform_final_bn = True

    def forward(self, value):
        value = self.a_relu(self.a_bn(self.a(value)))
        value = self.b_relu(self.b_bn(self.b(value)))
        return self.c_bn(self.c(value))


class I3DResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, inner, temporal_kernel, stride):
        super().__init__()
        if inputs != outputs or stride != 1:
            self.branch1 = nn.Conv3d(
                inputs,
                outputs,
                kernel_size=1,
                stride=(1, stride, stride),
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = nn.BatchNorm3d(outputs, eps=1e-5, momentum=0.1)
        self.branch2 = I3DBottleneck(inputs, outputs, inner, temporal_kernel, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value):
        # Evaluate the projected skip first, as in the original expression.
        if hasattr(self, "branch1"):
            value = self.branch1_bn(self.branch1(value)) + self.branch2(value)
        else:
            value = value + self.branch2(value)
        return self.relu(value)


class I3DStage(nn.Module):
    def __init__(self, inputs, outputs, inner, kernels, stride):
        super().__init__()
        self.blocks = len(kernels)
        for index, temporal in enumerate(kernels):
            self.add_module(
                f"pathway0_res{index}",
                I3DResidualBlock(
                    inputs if index == 0 else outputs,
                    outputs,
                    inner,
                    temporal,
                    stride if index == 0 else 1,
                ),
            )

    def forward(self, value):
        for index in range(self.blocks):
            value = getattr(self, f"pathway0_res{index}")(value)
        return value


class I3DProbabilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pathway0_avgpool = nn.AvgPool3d((8, 7, 7), stride=1)
        self.dropout = nn.Dropout(0.5)
        self.projection = nn.Linear(2048, 1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, value):
        # Retain the native single-path concatenation and NTHWC dropout layout.
        value = torch.cat([self.pathway0_avgpool(value)], dim=1)
        value = value.permute(0, 2, 3, 4, 1)
        value = self.act(self.projection(self.dropout(value)))
        # Larger features intentionally yield multiple outputs, never a mean.
        return value.view(value.shape[0], -1)


class I3DResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = SinglePathStem()
        self.s2 = I3DStage(64, 256, 64, (3, 3, 3), 1)
        self.pathway0_pool = nn.MaxPool3d((2, 1, 1), (2, 1, 1), padding=(0, 0, 0))
        self.s3 = I3DStage(256, 512, 128, (3, 1, 3, 1), 2)
        self.s4 = I3DStage(512, 1024, 256, (3, 1, 3, 1, 3, 1), 2)
        self.s5 = I3DStage(1024, 2048, 512, (1, 3, 1), 2)
        self.head = I3DProbabilityHead()
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                if module.weight is not None:
                    module.weight.data.fill_(
                        0.0 if getattr(module, "transform_final_bn", False) else 1.0
                    )
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.01)
                module.bias.data.zero_()

    def features(self, images):
        value = self.pathway0_pool(self.s2(self.s1(images)))
        return self.s5(self.s4(self.s3(value)))

    def forward(self, images):
        return self.head(self.features(images))


def load_i3d_resnet50(model_id):
    return I3DResNet50()
