"""Independent five-domain LSDA model, preserving partial freeze and live targets."""

import torch
from torch import nn
from torch.nn import functional as F

from .augmentation import augment_domains, mixup_in_latent_space, cosine_similarity_loss


class LSDAFeatureModel(nn.Module):
    def __init__(self, encoders_f, encoder_c, student_encoder):
        super().__init__()
        self.num_domains, self.num_classes = 5, 5
        self.encoder_feat_dim, self.half_fingerprint_dim = 512, 256
        self.lr, self.do, self.pool = (
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.count = 0
        self.encoders_f = nn.ModuleList(encoders_f)
        self.encoder_c, self.student_encoder = encoder_c, student_encoder
        self.fc_weights = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 5),
        )
        self.binary_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 2),
        )
        self.cls_criterion = nn.CrossEntropyLoss()

    def real_fake_feature_extract(self, images):
        fake, content = [], []
        for domain in range(images.shape[1]):
            selected = images[:, domain]
            content.append(self.encoder_c(selected))
            if domain > 0:
                fake.append(self.encoders_f[domain - 1].features(selected))
        return torch.stack(fake).transpose(1, 0), torch.stack(content).transpose(1, 0)

    def forward(self, images, labels=None, *, inference=False):
        if inference:
            return self.binary_classifier(self.student_encoder.features(images))
        groups, domains, channels, height, width = images.shape
        fake, content = self.real_fake_feature_extract(images)
        augmented, mixed = augment_domains(fake), mixup_in_latent_space(fake)
        augmented = torch.cat([augmented, mixed], dim=2).view(-1, 1024, 8, 8)
        target_fake = self.fc_weights(augmented).view(groups, domains - 1, 512, 8, 8)
        real, deepfakes, face2face, faceswap, textures = (
            content[:, index] for index in range(5)
        )
        constraint = (
            cosine_similarity_loss(real, textures).sum()
            + cosine_similarity_loss(real, face2face).sum()
            - cosine_similarity_loss(real, faceswap).sum()
            - cosine_similarity_loss(real, deepfakes).sum()
        )
        constraint = constraint.mean()
        labels = labels.contiguous().view(-1)
        binary = torch.where(labels == 0, 0, 1)
        student = self.student_encoder.features(
            images.view(-1, channels, height, width)
        )
        logits = self.binary_classifier(student)
        deepfake_loss = F.cross_entropy(logits, binary)
        real_mask, fake_mask = labels == 0, labels > 0
        real_student = student[real_mask]
        fake_student = student[fake_mask].reshape(groups, domains - 1, 512, 8, 8)
        real_distillation = F.mse_loss(
            real_student, content.reshape(-1, 512, 8, 8)[real_mask]
        )
        fake_distillation = F.mse_loss(fake_student, target_fake)
        distillation = real_distillation + fake_distillation
        all_domains = torch.cat([content[:, 0].unsqueeze(1), fake], dim=1).reshape(
            groups * domains, 512, 8, 8
        )
        domain_loss = self.cls_criterion(self.mlp(all_domains), labels)
        return logits, deepfake_loss, distillation, domain_loss, constraint, student


class LSDADetector(nn.Module):
    def __init__(self, encoders_f, encoder_c, student_encoder):
        super().__init__()
        self.model = LSDAFeatureModel(encoders_f, encoder_c, student_encoder)
        self.cls_criterion, self.gan_loss_fn = nn.CrossEntropyLoss(), nn.BCELoss()
        self.prob, self.label, self.correct, self.total = [], [], 0, 0

    def forward_outputs(self, images, labels=None, *, inference=False):
        result = self.model(images, labels, inference=inference)
        if inference:
            logits = result
            probability = logits.softmax(1)[:, 1]
            output = {"cls": logits, "prob": probability, "feat": probability}
            if labels is not None:
                self.prob.append(probability.detach().squeeze().cpu().numpy())
                self.label.append(labels.detach().squeeze().cpu().numpy())
                self.correct += (logits.argmax(1) == labels).sum().item()
                self.total += labels.size(0)
            return output
        logits, deepfake, distillation, domain, real, student = result
        return {
            "cls": logits,
            "prob": logits.softmax(1)[:, 1],
            "feat": student,
            "pred_loss": (deepfake, distillation, domain, real),
        }

    @staticmethod
    def loss_terms(output):
        # The original broad except hides malformed training outputs as zero.
        # Keep valid mathematical outputs, diagnose malformed training transport.
        if "pred_loss" not in output or len(output["pred_loss"]) != 4:
            raise ValueError("LSDA training requires all four native loss components")
        deepfake, distillation, domain, real = output["pred_loss"]
        return {
            "overall": 1 * domain + 0.5 * deepfake + 1 * distillation + 1 * real,
            "domain": domain,
            "deepfake": deepfake,
            "distillation": distillation,
            "real_loss": real,
        }

    def forward(self, images):
        return self.forward_outputs(images, inference=True)["cls"]
