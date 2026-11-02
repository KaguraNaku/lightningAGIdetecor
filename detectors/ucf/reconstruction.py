"""Native shared style/content reconstruction, including repeated layer use."""

from torch import nn


def double_conv(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    @staticmethod
    def c_norm(value, batch, channels, eps):
        variance = value.var(dim=-1) + eps
        std = variance.sqrt().view(batch, channels, 1, 1)
        mean = value.mean(dim=-1).view(batch, channels, 1, 1)
        return std, mean

    def forward(self, value, style):
        assert value.size(0) == style.size(0)
        size = value.size()
        batch, channels = size[:2]
        std, mean = self.c_norm(
            value.view(batch, channels, -1), batch, channels, self.eps
        )
        style_std, style_mean = self.c_norm(
            style.reshape(batch, channels, -1), batch, channels, self.eps
        )
        return ((value - mean.expand(size)) / std.expand(size)) * style_std.expand(
            size
        ) + style_mean.expand(size)


class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.adain3, self.adain2, self.adain1 = AdaIN(), AdaIN(), AdaIN()
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.activation = nn.Tanh()

    def forward(self, style, value):
        value = self.adain3(value, style)
        value = self.dconv_up3(self.dropout(self.upsample(value)))
        style = self.dconv_up3(self.dropout(self.upsample(style)))
        value = self.adain2(value, style)
        value = self.dconv_up2(self.dropout(self.upsample(value)))
        style = self.dconv_up2(self.dropout(self.upsample(style)))
        value = self.adain1(value, style)
        value = self.dconv_up1(self.dropout(self.upsample(value)))
        return self.activation(self.up_last(self.conv_last(value)))
