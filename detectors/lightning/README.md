# Lightning

Lightning fits an analytic readout on normalized global and patch-mean
features from selected blocks of a frozen vision backbone.

| Module | Object | Responsibility |
|---|---|---|
| `task.py` | `LightningTask` | Source fitting, independent validation, prediction and checkpoint state |
| `encoder.py` | `VisionEncoder` | Global and patch-mean features from frozen vision blocks |
| `detector.py` | `LightningDetector` | Equal raw-score fusion without target calibration |
| `model.py` | `AnalyticReadout` | Fixed maps, sufficient-statistic updates and analytic solutions |

`SpectralMapping`, `SufficientStatistics` and `MappingAccumulator` are the small
reusable numerical objects used by `AnalyticReadout`. Fitting streams moments;
it does not retain a new per-image feature matrix. The FF++ source statistics
are included once. Incremental source blocks inherit the selected parent head
and its statistics without replaying old image IDs.

Use `engine.py fit` with one of the three Lightning YAML presets under
`configs/experiments/`. FF++ can fit its mapping from the source, or use the
frozen paper mapping via `model.mapping_path`. A fresh UFD fit learns its own
mapping. For incremental fitting, `model.checkpoint` points to the
previous fit's `model.pth`. For continuation, declare only `factory` and
`checkpoint` under `model`. The checkpoint supplies the backbone, layers,
mapping and Ridge strength. The optional `dataset` field on the checkpoint
reference checks its training source.

Set `model.backbone_id` to a Hugging Face ID or local model directory. CLIP-L/14
is the default. Set `model.layers` to the zero-based transformer blocks to
read, for example `[0, 3, 7, 11]` for a 12-block backbone. CLIP, DINOv2 and
ViT use CLS for the global feature. SigLIP uses its pretrained attention pool.
Set `model.revision` to select a specific Hugging Face snapshot. Fit a new
mapping when changing the backbone or layers. The checkpoint records
the encoder, selected layers, and processor settings for evaluation and
continued fitting. Use `HF_HOME` and `HF_HUB_OFFLINE` for Hugging Face cache
and offline access.

The encoder uses FP32 by default. Use `--precision fp16` or `--precision bf16`
for mixed precision. CUDA allows TF32 by default for Lightning. Use
`--precision fp32 --no-tf32` for paper reproduction. Ridge statistics,
solving, and scores use FP64.

Each branch minimizes `||Zw - (y - mean(y))||² / n + λ||w||²`, where `Z` holds
the projected features. Both branches use `model.ridge_lambda` (default 0.15).
`engine.py evaluate` uses the fitted `model.pth`, which stores the maps,
sufficient statistics and folded coefficients.
