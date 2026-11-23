"""Device selection and backend-specific PyTorch settings."""

import torch


def available_accelerator():
    return (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else None
    )


def resolve_device(request):
    if request == "auto":
        return available_accelerator() or torch.device("cpu")
    device = torch.device(request)
    if device.type == "cpu":
        return device
    current = available_accelerator()
    if current is None or current.type != device.type:
        raise ValueError(f"Requested accelerator {device.type!r} is unavailable")
    if device.index is not None and device.index >= torch.accelerator.device_count():
        raise ValueError(f"Accelerator device index is out of range: {device}")
    return device


def validate_precision(device, precision):
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("Use fp32 or bf16 on CPU")
    if precision != "fp32" and not torch.amp.is_autocast_available(device.type):
        raise ValueError(f"Autocast is unavailable on {device.type}")
    if device.type == "cuda" and precision == "bf16":
        index = (
            device.index
            if device.index is not None
            else torch.accelerator.current_device_index()
        )
        with torch.accelerator.device_index(index):
            supported = torch.cuda.is_bf16_supported()  # noqa: TID251 - CUDA capability query
        if not supported:
            raise ValueError(f"CUDA device {device} does not support bf16")


def get_device_rng_state(device):
    backend = torch.get_device_module(device)
    return (
        backend.get_rng_state()
        if device.type == "mps"
        else backend.get_rng_state(device)
    )


def set_device_rng_state(device, state):
    backend = torch.get_device_module(device)
    if device.type == "mps":
        backend.set_rng_state(state)
    else:
        backend.set_rng_state(state, device)


def tf32_flags():
    matmul = torch.backends.cuda.matmul.allow_tf32  # noqa: TID251 - CUDA-only precision setting
    convolution = torch.backends.cudnn.allow_tf32  # noqa: TID251 - cuDNN-only precision setting
    return matmul, convolution


def set_tf32(matmul, convolution):
    torch.backends.cuda.matmul.allow_tf32 = matmul  # noqa: TID251 - CUDA-only precision setting
    torch.backends.cudnn.allow_tf32 = convolution  # noqa: TID251 - cuDNN-only precision setting


def cuda_metadata(device):
    if device.type != "cuda":
        return {}
    matmul, convolution = tf32_flags()
    return {
        "gpu": torch.cuda.get_device_name(device),  # noqa: TID251 - CUDA device metadata
        "matmul_allow_tf32": matmul,
        "cudnn_allow_tf32": convolution,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,  # noqa: TID251 - cuDNN metadata
        "cudnn_deterministic": torch.backends.cudnn.deterministic,  # noqa: TID251 - cuDNN metadata
    }
