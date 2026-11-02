"""Independent native HRNet-W48, including its unused classification-head state.

Selected architecture follows the local DeepfakeBench Microsoft HRNet variant.
Only Torch is imported; construction performs no checkpoint or configuration IO.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inputs, width, *, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, width, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, images):
        residual = images
        value = self.relu(self.bn1(self.conv1(images)))
        value = self.bn2(self.conv2(value))
        if self.downsample is not None:
            residual = self.downsample(images)
        value += residual
        return self.relu(value)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inputs, width, *, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, momentum=0.1)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width, momentum=0.1)
        self.conv3 = nn.Conv2d(width, width * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, images):
        residual = images
        value = self.relu(self.bn1(self.conv1(images)))
        value = self.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        if self.downsample is not None:
            residual = self.downsample(images)
        value += residual
        return self.relu(value)


def residual_stack(block, inputs, width, count, *, stride=1):
    downsample = None
    if stride != 1 or inputs != width * block.expansion:
        # Native constructor initializes this projection BEFORE the first block.
        downsample = nn.Sequential(
            nn.Conv2d(inputs, width * block.expansion, 1, stride, bias=False),
            nn.BatchNorm2d(width * block.expansion, momentum=0.1),
        )
    layers = [block(inputs, width, stride=stride, downsample=downsample)]
    layers.extend(block(width * block.expansion, width) for _ in range(count - 1))
    return nn.Sequential(*layers)


class ResolutionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.num_branches = len(channels)
        self.branches = nn.ModuleList(
            residual_stack(BasicBlock, width, width, 4) for width in channels
        )
        rows = []
        for target, target_width in enumerate(channels):
            transforms = []
            for source, source_width in enumerate(channels):
                if source == target:
                    transforms.append(None)
                elif source > target:
                    transforms.append(
                        nn.Sequential(
                            nn.Conv2d(source_width, target_width, 1, bias=False),
                            nn.BatchNorm2d(target_width, momentum=0.1),
                            nn.Upsample(
                                scale_factor=2 ** (source - target), mode="nearest"
                            ),
                        )
                    )
                else:
                    steps = []
                    for index in range(target - source):
                        last = index == target - source - 1
                        outputs = target_width if last else source_width
                        layers = [
                            nn.Conv2d(source_width, outputs, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outputs, momentum=0.1),
                        ]
                        if not last:
                            layers.append(nn.ReLU(inplace=False))
                        steps.append(nn.Sequential(*layers))
                    transforms.append(nn.Sequential(*steps))
            rows.append(nn.ModuleList(transforms))
        self.fuse_layers = nn.ModuleList(rows)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, values):
        for index, branch in enumerate(self.branches):
            values[index] = branch(values[index])
        outputs = []
        for target, transforms in enumerate(self.fuse_layers):
            value = values[0] if target == 0 else transforms[0](values[0])
            for source in range(1, self.num_branches):
                value = value + (
                    values[source]
                    if target == source
                    else transforms[source](values[source])
                )
            outputs.append(self.relu(value))
        return outputs


def transition(previous, current):
    layers = []
    for branch, width in enumerate(current):
        if branch < len(previous):
            layers.append(
                None
                if previous[branch] == width
                else nn.Sequential(
                    nn.Conv2d(previous[branch], width, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(width, momentum=0.1),
                    nn.ReLU(inplace=True),
                )
            )
        else:
            steps = []
            for index in range(branch + 1 - len(previous)):
                outputs = width if index == branch - len(previous) else previous[-1]
                steps.append(
                    nn.Sequential(
                        nn.Conv2d(previous[-1], outputs, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outputs, momentum=0.1),
                        nn.ReLU(inplace=True),
                    )
                )
            layers.append(nn.Sequential(*steps))
    return nn.ModuleList(layers)


class HRNetW48(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = residual_stack(Bottleneck, 64, 64, 4)
        self.transition1 = transition([256], [48, 96])
        self.stage2 = nn.Sequential(ResolutionFusion([48, 96]))
        self.transition2 = transition([48, 96], [48, 96, 192])
        self.stage3 = nn.Sequential(
            *(ResolutionFusion([48, 96, 192]) for _ in range(4))
        )
        self.transition3 = transition([48, 96, 192], [48, 96, 192, 384])
        self.stage4 = nn.Sequential(
            *(ResolutionFusion([48, 96, 192, 384]) for _ in range(3))
        )
        # FaceXray uses features(), but these parameters consume constructor RNG
        # and remain in the source state/trainable inventory without gradients.
        self.incre_modules = nn.ModuleList(
            residual_stack(Bottleneck, inputs, width, 1)
            for inputs, width in zip(
                [48, 96, 192, 384], [32, 64, 128, 256], strict=True
            )
        )
        self.downsamp_modules = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, 2, 1),
                nn.BatchNorm2d(outputs, momentum=0.1),
                nn.ReLU(inplace=True),
            )
            for inputs, outputs in [(128, 256), (256, 512), (512, 1024)]
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(1024, 2048, 1),
            nn.BatchNorm2d(2048, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(2048, 1000)

    def resolutions(self, images):
        value = self.relu(self.bn1(self.conv1(images)))
        value = self.relu(self.bn2(self.conv2(value)))
        value = self.layer1(value)
        values = [
            layer(value) if layer is not None else value for layer in self.transition1
        ]
        values = self.stage2(values)
        for transforms, stage in (
            (self.transition2, self.stage3),
            (self.transition3, self.stage4),
        ):
            values = stage(
                [
                    layer(values[-1]) if layer is not None else values[index]
                    for index, layer in enumerate(transforms)
                ]
            )
        return values

    def features(self, images):
        values = self.resolutions(images)
        size = values[0].shape[2:]
        return torch.cat(
            [values[0]]
            + [
                F.interpolate(value, size=size, mode="bilinear") for value in values[1:]
            ],
            dim=1,
        )

    def forward(self, images):
        values = self.resolutions(images)
        value = self.incre_modules[0](values[0])
        for index, downsample in enumerate(self.downsamp_modules):
            value = self.incre_modules[index + 1](values[index + 1]) + downsample(value)
        value = self.final_layer(value)
        if torch._C._get_tracing_state():
            value = value.flatten(start_dim=2).mean(dim=2)
        else:
            value = F.avg_pool2d(value, value.size()[2:]).view(value.size(0), -1)
        return self.fc(value)


def load_hrnet_w48(model_id, **kwargs):
    if kwargs:
        raise TypeError(
            f"Native HRNetW48 has no architecture overrides: {sorted(kwargs)}"
        )
    return HRNetW48()
