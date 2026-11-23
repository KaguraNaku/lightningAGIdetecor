# Lightning

Lightning fits an analytic readout on normalized global and patch-mean
features from selected blocks of a frozen vision backbone.

| Module | Object | Responsibility |
|---|---|---|
| `task.py` | `LightningTask` | Fitting, resume, validation and prediction |
| `config.py` | | Defaults for the backbone, selected layers, projection rank and Ridge strength |
| `encoder.py` | `VisionEncoder` | Global and patch-mean features from frozen vision blocks |
| `detector.py` | `LightningDetector` | Equal raw-score fusion without target calibration |
| `model.py` | `AnalyticReadout` | Fixed maps, sufficient-statistic updates and analytic solutions |

`SpectralMapping`, `SufficientStatistics` and `MappingAccumulator` are the small
reusable numerical objects used by `AnalyticReadout`. Fitting accumulates
moments batch by batch. Continued fitting starts with the saved statistics
and rejects repeated training image IDs.

Use `engine.py fit --model lightning` with the shared
[`configs/config.yaml`](../../configs/config.yaml). Select the training and
validation inputs with `--train-input` and `--validation-input`. A fresh fit
learns its mapping from the selected source. For the paper's FF++ mapping,
set `models.lightning.settings.mapping_path`.

Pass a fitted `model.pth` to `--checkpoint` to continue from its encoder,
mapping, Ridge strength and accumulated statistics. Use `--train-input` to
select the next set of images. The validation input provides metrics.

Edit [`config.py`](config.py) to change the backbone, layers, projection rank
or Ridge strength for new fits. Use a Hugging Face ID or an absolute path to
a local model directory for the backbone. Layers are zero-based, for
example `[0, 3, 7, 11]` for a 12-block backbone. The supported Hugging Face
architectures are CLIP, DINOv2, ViT and SigLIP. CLIP, DINOv2 and ViT use CLS for
the global feature. SigLIP uses its pretrained attention pool.
Set `BACKBONE_REVISION` to select a specific Hugging Face snapshot. Fit a new
mapping when changing the backbone or layers. The checkpoint records the
encoder, selected layers, processor settings and fitted mapping for evaluation
and continued fitting. Use `HF_HOME` and `HF_HUB_OFFLINE` for Hugging Face cache
and offline access.

The encoder uses FP32 by default. Use `--precision fp16` or `--precision bf16`
for mixed precision. CUDA allows TF32 by default for Lightning. Use
`--precision fp32 --no-tf32` for paper reproduction. Ridge statistics,
solving, and scores use FP64.

Each branch minimizes `||Zw - (y - mean(y))||² / n + λ||w||²`, where `Z` holds
the projected features. Both branches share the same Ridge strength.
`engine.py evaluate` uses the fitted `model.pth`, which stores the maps,
sufficient statistics and folded coefficients.
Checkpoint statistics use the `SufficientStatistics` attribute names. Use the
original Git revision to read checkpoints with abbreviated statistic keys.
