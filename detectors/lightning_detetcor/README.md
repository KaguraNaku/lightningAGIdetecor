# Fixed-layer Global + Mean pipeline

The current pipeline preserves `LightningDetector(backbone, mapping, classifier)`.
Use the dual-view encoder and the explicit calibration/restoration entry points:

```python
from detectors.lightning_detetcor.encoder_factory import load_encoder
from detectors.lightning_detetcor.calibration import (
    calibrate_global_mean, load_global_mean_detector, partial_fit,
)

encoder = load_encoder("openai/clip-vit-large-patch14", readout="global_mean")
config = dict(layers_count=24, input_dim=1024, output_dim=128,
              topk=8, eps=1e-6, ridge_lambda=0.15)
# Source features: {"global": tensor(N,24,1024), "mean": tensor(N,24,1024)}.
# Extract/cache in batches as needed; use this checkpoint's native image processor.
detector = calibrate_global_mean(encoder, source_features, source_labels, config)
scores = detector(processed_images)  # higher = fake; raw sum of both heads
state = detector.state_dict()
# Restore with a fresh encoder of the same model/readout/revision.
restored = load_global_mean_detector(fresh_encoder, config, state)
partial_fit(restored, new_source_features, new_source_labels)
```

For CLIP, global means the L2-normalized hidden CLS token at each block, before
the final image projection; mean is the L2-normalized mean of patch tokens.
For SigLIP, global uses the adapter's frozen native attention pool instead.
Eight fixed layers cover blocks **1, 4, 8, 11, 14, 17, 21, 24** (one-based).
Each branch/layer independently estimates a real-only mean and population
variance, uses `sqrt(variance + eps)` for scaling, and keeps the leading 128
eigenvectors of the normalized features' **uncentered second moment**. There is
no pooled re-centering. Both source classes enter that second moment and the
supervised Ridge fit. Each branch has its own FP64 Ridge statistics, centered
binary targets, no free intercept, and a block-trace penalty. Scores are added
with equal weights; no LDA, score calibration, Std, or learned layer ranking.

Online updates keep the encoder, selected layers and feature maps fixed. Both
heads are solved before either update is committed. Checkpoints preserve the
statistics and verify per-view mapping fingerprints before reuse. Mapping/Ridge
buffers default to the feature device during calibration and can be on CPU
independently of the encoder. This is in-memory calibration, not a streaming
feature extractor. Legacy single-view `calibrate` / `load_detector`, layer search
and LDA remain available for old checkpoints and comparisons.

Dataset-specific experiments, reports, raw predictions and weights belong in
the paper workspace or external artifact storage, not in this code repository.
