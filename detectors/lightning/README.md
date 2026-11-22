# Lightning

The maintained implementation uses the frozen CLIP encoder, eight declared
transformer layers, normalized CLS and patch-mean readouts, and equal raw-score
analytic heads. It does not search layers, use LDA, or choose a head by target
performance. Source validation is reported separately from fitting.

| Module | Object | Responsibility |
|---|---|---|
| `task.py` | `LightningTask` | Source fitting, independent validation, prediction and checkpoint state |
| `encoder.py` | `CLIPEncoder` | Frozen per-block CLIP readouts, loaded only when needed |
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
previous fit's `model.pth`. The new source block is added to its statistics
using the same mapping and Ridge strength. The optional `dataset` field on
the checkpoint reference checks its training source. Use the Hugging Face
`HF_HOME` and `HF_HUB_OFFLINE` settings for cache and offline access.

CLIP uses FP32 by default. Use `--precision fp16` or `--precision bf16`
for mixed precision. CUDA allows TF32 by default for Lightning. Use
`--precision fp32 --no-tf32` for paper reproduction. Ridge statistics,
solving, and scores use FP64.

Each branch minimizes `||Zw - (y - mean(y))||² / n + λ||w||²`, where `Z` holds
the projected features. Both branches use `model.ridge_lambda` (default 0.15).
`engine.py evaluate` uses the fitted `model.pth`, which stores the maps,
sufficient statistics and folded coefficients.
