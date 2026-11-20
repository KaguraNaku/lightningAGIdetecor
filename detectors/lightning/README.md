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
mapping. An incremental UFD stage restores the previous `model.pth` and adds
the new source block to the inherited statistics.

`engine.py evaluate` uses the fitted `model.pth`. It stores the fixed maps,
sufficient statistics and folded coefficients from uniform mean-loss ridge,
with `model.ridge_lambda` defaulting to 0.15. The frozen CLIP weights are shared.
