# 骨干模型加载

`BackboneRegistry` 保存加载函数的 import 路径或调用对象。`BackboneLoader` 只在调用时导入选中的模块。`load_backbone` 接受显式 `model_type`，也可从 Hugging Face 配置读取类型。缓存、revision、offline 和 dtype 参数按原接口传给选中的 loader。

| 归属 | 代码路径 |
|---|---|
| CLIP、DINOv2、ViT、SigLIP 通用 vision loader | `models/` |
| Xception，包括 CORE 和 F3Net 复用的实现 | `detectors/xception/backbone.py` |
| UCF 专用 Xception 和卷积块 | `detectors/ucf/` |
| ProDet 专用 EfficientNet-B4 和初始化 | `detectors/prodet/` |
| GenD 的 CLIP encoder | `detectors/gend/encoder.py` |
| ForAda 的原生 CLIP | `detectors/forada/clip.py` |

SigLIP loader 支持声明 `siglip` 或 `siglip_vision_model` 的 checkpoint。ViT loader 禁用 pooler。Lightning 从 `detectors/lightning/config.py` 选择 backbone，支持范围由 `encoder.py` 中的 readout 注册表确定。Checkpoint 记录 backbone 和 processor 配置。
