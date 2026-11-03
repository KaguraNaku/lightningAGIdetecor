# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted from DeepfakeBench detectors/utils/prodet_api.py.
"""Native CPU Beta draws andordered adjacent-domain latent mixing."""

import torch


def mixup_in_latent_space(domain2, domain1):
    if (
        domain1.shape != domain2.shape
        or domain1.dtype != torch.float32
        or domain2.dtype != torch.float32
    ):
        raise ValueError("ProDet mixup requires equal FP32 domain feature shapes")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Native ProDet mixing requires default float32")
    bs = domain1.shape[0]
    mixed_data = torch.zeros_like(domain1)
    # Native ratios andrandom draws originate on CPU,including when features
    # are CUDA tensors. Do not move the random draws to the feature device.
    ratios = torch.zeros(bs, device="cpu")
    for i in range(bs):
        alpha = torch.rand(1, device="cpu") * 1.5 + 0.5
        lambda_cpu = torch.distributions.beta.Beta(alpha, alpha).sample()
        lambda_ = lambda_cpu.to(domain1.device)
        mixed_data[i] = lambda_ * domain1[i] + (1 - lambda_) * domain2[i]
        # Native copies the same scalar back from CUDA to this CPU output.
        ratios[i] = lambda_cpu
    return mixed_data, ratios


def feature_operation_v2(features):
    if (
        features.ndim != 4
        or tuple(features.shape[1:]) != (512, 8, 8)
        or not len(features)
        or len(features) % 4
    ):
        raise ValueError("ProDet requires four equal ordered512x8x8 feature groups")
    single_len = len(features) // 4
    real = features[:single_len]
    blend = features[single_len : single_len * 2]
    bi = features[single_len * 2 : single_len * 3]
    fake = features[single_len * 3 : single_len * 4]
    real_blends, rb_ratios = mixup_in_latent_space(real, blend)
    blend_bis, bb_ratios = mixup_in_latent_space(blend, bi)
    bi_fakes, bf_ratios = mixup_in_latent_space(bi, fake)
    return real_blends, blend_bis, bi_fakes, rb_ratios, bb_ratios, bf_ratios
