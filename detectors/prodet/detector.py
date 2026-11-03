# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from DeepfakeBench ProDet_detector.py;selected linear native recipe.
"""ProDet projection,three experts,attention,drift andfinal binary classifier.

The backbone is injected. Model mathematics do not own epoch loops or source
dataset routing;the caller supplies the zero-based native epoch explicitly.
"""

import torch
from torch import nn

from .components import DriftBlock, FeatureAttentionBlock
from .losses import SoftCrossEntropyLoss
from .operations import feature_operation_v2


def binary_head():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 2),
    )


class ProDetDetector(nn.Module):
    def __init__(
        self, backbone, *, feature_drift=True, feature_operation=True, op_start=0
    ):
        super().__init__()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError(
                "The selected native ProDet profile requires default float32"
            )
        if type(feature_drift) is not bool or type(feature_operation) is not bool:
            raise ValueError("ProDet feature switches must be boolean")
        if type(op_start) is not int or op_start < 0:
            raise ValueError(
                "ProDet operation start must be a nonnegative native epoch"
            )
        self.feature_drift_enabled = feature_drift
        self.feature_operation_enabled = feature_operation
        self.op_start = op_start
        # Native build_backbone constructs the backbone first,but registers
        # these heads on the Detector before assigning the returned backbone.
        # Preserve both random-draw order andparameter/optimizer ordering.
        self.adjust_feature = nn.Conv2d(1792, 512, 1)
        self.blend_classifier = binary_head()
        self.fea_att = FeatureAttentionBlock()
        self.ID_inconsistency_classifier = binary_head()
        self.deepfake_classifier = binary_head()
        self.final_classifier = binary_head()
        self.backbone = backbone
        self.fea_loss = nn.MSELoss()
        self.loss_func = SoftCrossEntropyLoss()
        self.real2sbi = DriftBlock()
        self.sbi2bi = DriftBlock()
        self.bi2fake = DriftBlock()

    def classifier(self, features):
        df_pred = self.deepfake_classifier(features)
        bld_pred = self.blend_classifier(features)
        bi_pred = self.ID_inconsistency_classifier(features)
        return df_pred, bld_pred, bi_pred

    def feature_drift(self, features):
        single_len = len(features) // 4
        real, blend, bi, fake = (
            features[:single_len],
            features[single_len : single_len * 2],
            features[single_len * 2 : single_len * 3],
            features[single_len * 3 :],
        )
        real2blend = self.real2sbi(real)
        blend2bi = self.sbi2bi(blend)
        bi2fake = self.bi2fake(bi)
        # Keep target gradients exactly as native;these features are not detached.
        return real2blend, blend2bi, bi2fake, blend, bi, fake

    def forward_from_features(self, raw_features, *, inference, epoch=None):
        if (
            raw_features.ndim != 4
            or tuple(raw_features.shape[1:]) != (1792, 8, 8)
            or raw_features.dtype != torch.float32
        ):
            raise ValueError("ProDet requires full FP32 B4 features1792x8x8")
        if type(inference) is not bool:
            raise ValueError("ProDet inference flag must be boolean")
        if not inference and (not len(raw_features) or len(raw_features) % 4):
            raise ValueError("ProDet training needs four equal ordered domain groups")
        if not inference and (type(epoch) is not int or epoch < 0):
            raise ValueError("ProDet training requires its zero-based native epoch")
        features = self.adjust_feature(raw_features)
        pred_dict = {}
        if not inference:
            if self.feature_drift_enabled:
                r2b, b2b, b2f, bgt, bigt, fgt = self.feature_drift(features)
                pred_dict.update(r2b=r2b, b2b=b2b, b2f=b2f, bgt=bgt, bigt=bigt, fgt=fgt)
            if self.feature_operation_enabled and epoch >= self.op_start:
                real_blends, blend_bis, bi_fakes, rb_ratios, bb_ratios, bf_ratios = (
                    feature_operation_v2(features)
                )
                pred_dict.update(
                    rb_ratios=rb_ratios, bb_ratios=bb_ratios, bf_ratios=bf_ratios
                )
                features = torch.cat(
                    (features, real_blends, blend_bis, bi_fakes), dim=0
                )
        df_pred, bld_pred, bi_pred = self.classifier(features)
        att_features = self.fea_att(df_pred, bld_pred, bi_pred, features)
        pred = self.final_classifier(att_features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict.update(
            cls=pred,
            prob=prob,
            df_pred=df_pred,
            bld_pred=bld_pred,
            bi_pred=bi_pred,
            feat=features,
            att_features=att_features,
        )
        return pred_dict

    def forward_outputs(self, images, *, inference=True, epoch=None):
        if (
            images.ndim != 4
            or tuple(images.shape[1:]) != (3, 256, 256)
            or images.dtype != torch.float32
        ):
            raise ValueError("ProDet requires NCHW RGB256 FP32 inputs")
        return self.forward_from_features(
            self.backbone.features(images), inference=inference, epoch=epoch
        )

    def forward(self, images):
        return self.forward_outputs(images, inference=True)["cls"]

    @staticmethod
    def generate_labels(label, pred_dict):
        blend_label = torch.clip(label, 0, 1)
        # CPU integer scalars are copied to the label device,as native .cuda().
        one = torch.tensor(1, device="cpu").to(label.device)
        zero = torch.tensor(0, device="cpu").to(label.device)
        df_label = torch.where(label == 2, one, zero)
        id_diff_label = torch.where(label >= 2, one, zero)
        if "rb_ratios" in pred_dict:
            rb_label = pred_dict["rb_ratios"].to(label.device)
            bb_label = pred_dict["bb_ratios"].to(label.device)
            bf_label = pred_dict["bf_ratios"].to(label.device)
            blend_label = torch.cat(
                (
                    blend_label,
                    rb_label,
                    torch.ones_like(bb_label),
                    torch.ones_like(bf_label),
                ),
                dim=0,
            )
            id_diff_label = torch.cat(
                (
                    id_diff_label,
                    torch.zeros_like(rb_label),
                    bb_label,
                    torch.ones_like(bf_label),
                ),
                dim=0,
            )
            df_label = torch.cat(
                (
                    df_label,
                    torch.zeros_like(rb_label),
                    torch.zeros_like(bb_label),
                    bf_label,
                ),
                dim=0,
            )
        return blend_label, df_label, id_diff_label

    def get_drift_loss(self, pred_dict):
        if "r2b" in pred_dict:
            r2b_loss = self.fea_loss(pred_dict["r2b"], pred_dict["bgt"])
            b2bi_loss = self.fea_loss(pred_dict["b2b"], pred_dict["bigt"])
            b2f_loss = self.fea_loss(pred_dict["b2f"], pred_dict["fgt"])
        else:
            r2b_loss, b2bi_loss, b2f_loss = 0, 0, 0
        return r2b_loss, b2bi_loss, b2f_loss

    def loss_terms(self, pred_dict, labels):
        if labels.ndim != 1 or labels.dtype not in {torch.int64, torch.float32}:
            raise ValueError(
                "ProDet requires native int64 or FP32 domain/soft labels in[0,3]"
            )
        if not labels.is_meta and (
            not torch.isfinite(labels).all()
            or not ((labels >= 0) & (labels <= 3)).all()
        ):
            raise ValueError(
                "ProDet requires native int64 or FP32 domain/soft labels in[0,3]"
            )
        bld_label, df_label, id_diff_label = self.generate_labels(labels, pred_dict)
        loss_df = self.loss_func(pred_dict["df_pred"], df_label)
        loss_bld = self.loss_func(pred_dict["bld_pred"], bld_label)
        loss_idd = self.loss_func(pred_dict["bi_pred"], id_diff_label)
        if self.feature_drift_enabled:
            r2b_loss, b2b_loss, b2f_loss = self.get_drift_loss(pred_dict)
            loss = loss_bld + loss_df + loss_idd + r2b_loss + b2b_loss + b2f_loss
            loss_dict = {
                "overall": loss,
                "loss_df": loss_df,
                "loss_bld": loss_bld,
                "loss_iid": loss_idd,
                "r2b_loss": r2b_loss,
                "b2b_loss": b2b_loss,
                "b2f_loss": b2f_loss,
            }
        else:
            loss = loss_bld + loss_df + loss_idd
            loss_dict = {
                "overall": loss,
                "loss_df": loss_df,
                "loss_bld": loss_bld,
                "loss_iid": loss_idd,
            }
        loss_out = self.loss_func(pred_dict["cls"], bld_label)
        loss += loss_out
        loss_dict["loss_linear"] = loss_out
        return loss_dict
