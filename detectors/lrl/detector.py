"""Native dual-B4 relation learning, retaining literal attention/decoder math."""

import torch
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.InstanceNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.layer(value)


class ConcatBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.seq(value)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_filters, self.filters = (
            [64, 128, 256, 256],
            [48, 40, 64, 176, 2008],
        )
        # Native construction is forward order, execution is reverse order.
        self.decoder_stages = nn.ModuleList(
            [
                DecoderBlock(inputs, outputs)
                for inputs, outputs in [(128, 64), (256, 128), (256, 256), (2008, 256)]
            ]
        )
        self.bottlenecks = nn.ModuleList(
            [ConcatBottleneck(channels) for channels in [256, 256, 128, 64]]
        )
        self.last_block = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, encoder_results):
        value = encoder_results[0]
        for index, bottleneck in enumerate(self.bottlenecks):
            value = bottleneck(self.decoder_stages[-(index + 1)](value))
        return self.last_block(value)


class RFAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 2, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb, frequency):
        attention = self.attention(torch.cat((rgb, frequency), dim=1))
        first = attention[:, 0, ...].unsqueeze(1).contiguous()
        second = attention[:, 1, ...].unsqueeze(1).contiguous()
        rgb *= first
        frequency *= second
        return rgb, frequency


class LRLDetector(nn.Module):
    def __init__(self, encoder_rgb, encoder_idct):
        super().__init__()
        self.encoder_rgb, self.encoder_idct = encoder_rgb, encoder_idct
        # The source constructs a second3-channel stem before replacing it here.
        self.encoder_idct.efficientnet._conv_stem = nn.Conv2d(
            1, 48, 3, stride=2, bias=False
        )
        self.seg_loss, self.sim_loss = nn.BCELoss(), nn.MSELoss()
        self.loss_func = nn.CrossEntropyLoss()
        self.feature_adjust1, self.feature_adjust2 = (
            nn.Upsample(scale_factor=0.25),
            nn.Upsample(scale_factor=0.5),
        )
        self.decoder = Decoder()
        self.rfam1, self.rfam2, self.rfam3 = RFAM(56), RFAM(160), RFAM(1792)
        self.final = nn.Conv2d(64, 1, 1, bias=False)
        self.overall_classifier = nn.Sequential(
            nn.Linear(240, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        )

    def features(self, images, idct):
        rgb1, idct1 = self.rfam1(
            self.encoder_rgb.block_part1(images), self.encoder_idct.block_part1(idct)
        )
        low = rgb1 + idct1
        rgb2, idct2 = self.rfam2(
            self.encoder_rgb.block_part2(rgb1), self.encoder_idct.block_part2(idct1)
        )
        mid = rgb2 + idct2
        rgb3, idct3 = self.rfam3(
            self.encoder_rgb.block_part3(rgb2), self.encoder_idct.block_part3(idct2)
        )
        high = rgb3 + idct3
        return torch.cat(
            (self.feature_adjust1(low), self.feature_adjust2(mid), high), dim=1
        )

    def feature_process(self, features):
        patches = F.unfold(features, kernel_size=2, stride=2, padding=0).permute(
            0, 2, 1
        )
        normalized = patches / (patches * patches).sum(dim=2, keepdim=True).sqrt()
        batch, count = patches.shape[:2]
        similarity = torch.einsum(
            "bij,bjk->bik", normalized, normalized.permute(0, 2, 1)
        )
        similarity = (similarity + 1) / 2
        mask = (
            (torch.eye(count) != 1)
            .repeat(batch, 1)
            .view(batch, count, count)
            .to(features.device)
        )
        selected = torch.masked_select(similarity, mask).view(batch, count, -1)
        return selected.view(batch, -1), similarity

    @staticmethod
    def similarity_targets(masks):
        # Keep chunk/mean order and native detached CPU scalar construction;
        # adaptive pooling changes floating-point reductions on soft masks.
        targets = []
        for mask in masks:
            ratios = [
                part.float().mean()
                for row in torch.chunk(mask, 4, dim=0)
                for part in torch.chunk(row, 4, dim=1)
            ]
            ratios = torch.tensor(ratios, device="cpu").view(-1, 1)
            targets.append(1 - torch.norm(ratios[:, None] - ratios, dim=2, p=2))
        return torch.stack(targets).to(masks.device)

    def forward_outputs(self, images, idct):
        features = self.features(images, idct)
        relations, similarity = self.feature_process(features)
        logits = self.overall_classifier(relations)
        mask = torch.sigmoid(self.final(self.decoder([features])))
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": features,
            "mask_pred": mask,
            "sim": similarity,
        }

    def forward(self, images, idct):
        return self.forward_outputs(images, idct)["cls"]

    def loss_terms(self, outputs, labels, masks):
        target = self.similarity_targets(masks.squeeze(1))
        similarity = self.sim_loss(outputs["sim"], target)
        segmentation = self.seg_loss(outputs["mask_pred"], masks)
        classification = self.loss_func(outputs["cls"], labels)
        return {
            "overall": similarity + segmentation + classification,
            "sim_loss": similarity,
            "seg_loss": segmentation,
            "ce_loss": classification,
        }
