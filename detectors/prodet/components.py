# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from local DeepfakeBench ProDet_detector.py.
# Change: CPU-drawn noise moves to x.device instead of unconditional CUDA.
# Retain native constructor order,default initializers andall selected operators.
"""Native drift andfeature-attention modules;no benchmark framework imports."""

import torch
from torch import nn
from torch.nn import init


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class DriftBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_process = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.concat_process = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(inplace=True),
        )

    def conv_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def init_block(self):
        self.concat_process.apply(self.conv_init)
        self.noise_process.apply(self.conv_init)

    def forward(self, x):
        noise = torch.randn(x.shape[0], 512, 8, 8).to(x.device)
        noise_processed = self.noise_process(noise)
        trans_res = self.concat_process(torch.cat((x, noise_processed), dim=1))
        return trans_res


class FeatureAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inverse_conv1 = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8)),
        )
        self.inverse_conv2 = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8)),
        )
        self.inverse_conv3 = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),
            nn.ConvTranspose2d(512, 512, kernel_size=(8, 8)),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(512 * 4, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, x3, feature):
        x1 = self.inverse_conv1(x1)
        x2 = self.inverse_conv2(x2)
        x3 = self.inverse_conv3(x3)
        att = self.attention(torch.cat((x1, x2, x3, feature), dim=1))
        feature = feature * att
        return feature
