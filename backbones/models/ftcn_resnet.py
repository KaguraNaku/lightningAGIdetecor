"""Independent FTCN R50 conversion preserving native replacement/RNG order."""

from torch import nn

from .i3d_resnet import I3DResNet50
from .ftcn_temporal import FTCNTemporalHead


def convert_temporal(
    module, name="model", *, removed=0, stride_removed=0, receipt=None
):
    if receipt is None:
        receipt = []
    # The source scans attributes alphabetically BEFORE descending through the
    # now-mutated children. A named_modules sweep changes initialization order.
    for attribute in dir(module):
        child = getattr(module, attribute)
        if type(child) is nn.Conv3d:
            kernel = list(child.kernel_size)
            if kernel[1] != kernel[2]:
                raise ValueError("Native FTCN needs square spatial convolution kernels")
            stride = child.stride
            extra = None
            if stride[1] == stride[2] == 2:
                stride_removed += 1
                stride = [1, 1, 1]
                extra = nn.MaxPool3d((1, 2, 2))
            if kernel[1] == 1 and extra is None:
                continue
            padding = list(child.padding)
            kernel[1] = kernel[2] = 1
            padding[1] = padding[2] = 0
            # Native I3D convolutions are bias-free. The old code forwards the
            # .bias attribute (None), not a weight-copy operation.
            if child.bias is not None:
                raise ValueError(
                    "The selected FTCN I3D recipe requires bias-free convolutions"
                )
            replacement = nn.Conv3d(
                child.in_channels,
                child.out_channels,
                kernel,
                stride=stride,
                padding=padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=None,
                padding_mode=child.padding_mode,
            )
            setattr(module, attribute, replacement)
            removed += 1
            row = {
                "path": name + "." + attribute,
                "original_kernel": list(child.kernel_size),
                "kernel": kernel,
                "stride": list(stride),
                "fresh_random_weights": True,
                "stride_index": stride_removed,
            }
            if extra is not None:
                bn_name = "bn" if attribute == "conv" else attribute + "_bn"
                if not hasattr(module, bn_name) or not isinstance(
                    getattr(module, bn_name), nn.BatchNorm3d
                ):
                    raise ValueError(
                        "Native FTCN stride pool needs its associated BatchNorm"
                    )
                setattr(module, bn_name, nn.Sequential(getattr(module, bn_name), extra))
                row["post_BN_pool"] = name + "." + bn_name
            receipt.append(row)
        elif type(child) is nn.Dropout:
            setattr(module, attribute, nn.Dropout(0.5))
        # Source no_time_pool=True. These adjustments include the old head
        # that is discarded only after the complete conversion has run.
        if type(child) is nn.MaxPool3d:
            kernel = list(child.kernel_size)
            if kernel[0] == 2:
                kernel[0] = 1
                setattr(module, attribute, nn.MaxPool3d(kernel))
        elif type(child) is nn.AvgPool3d:
            kernel = list(child.kernel_size)
            kernel[0] *= 2
            setattr(module, attribute, nn.AvgPool3d(kernel))
    for attribute, child in module.named_children():
        removed, stride_removed = convert_temporal(
            child,
            name + "." + attribute,
            removed=removed,
            stride_removed=stride_removed,
            receipt=receipt,
        )
    return removed, stride_removed


class FTCNResNet50(I3DResNet50):
    def __init__(self, *, initial_state=None):
        super().__init__()
        if initial_state is not None:
            self.load_state_dict(initial_state, strict=True)
        self.conversion_receipt = []
        self.removed_convolutions, self.removed_strides = convert_temporal(
            self, receipt=self.conversion_receipt
        )
        # Do not skip initializing/replacing s5 earlier: its random draws are
        # part of the original constructor even though it is discarded here.
        self.s5 = nn.Identity()
        self.head = FTCNTemporalHead()


def load_ftcn_resnet50(model_id, **kwargs):
    if model_id != "ftcn_resnet50":
        raise ValueError("Expected the native ftcn_resnet50 identifier")
    return FTCNResNet50(**kwargs)
