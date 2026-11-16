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

The common experiment entry point selects `--model lightning`. Its maintained
factory is `detectors.lightning.task:LightningTask`. The same engine can read an
existing head directory containing `coefficients.npz` and `mapping.npz`, including
relative component symlinks. The pinned frozen encoder remains a shared asset;
small analytic checkpoints do not duplicate its weights.

The `compatibility/` package contains the older generic Lightning Task and math
needed to read historical checkpoints and run their existing parity tests. It is
not another model in the public catalog. Historical factory strings are resolved
by the common factory registry rather than alias files or `sys.modules` tricks.
The rename preserves the existing checkpoint format/version and parameter keys.
