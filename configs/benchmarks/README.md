# First requested detector batch

Branch: `codex/benchmark-eight-native`, based on committed
`codex/migrate-effort@240274d1dc865bb156b2c90f11fb9952b1a641f5`.
The source branch's uncommitted SLADD work is not imported or modified.
Only Xception, F3Net, CORE, UCF, ProDet, Effort, ForAda and GenD are active in
`first_batch.json`; inherited other detectors remain unchanged.

## Architecture and data

`engine -> method Task -> method Detector -> Backbone` remains the contract.
Shared manifests own image/video identity, source membership and labels.
Each method owns its preprocessing, auxiliary supervision, objective, freeze
policy and optimizer through composition or the existing trainer hooks.
The eight examples in `eight/` use shared training8 / validation32 / testing32
frame membership. This transport protocol is not the authors' full original
sampling protocol. Single images report FrameAUC; videos report VideoAUC from
mean frame probabilities. Source validation selects checkpoints; target tests
do not select models or settings.

The examples contain explicit initialization asset paths. They are templates,
not a claim that those weights/data are already installed or that a full
experiment was run. Invoke the existing engine CLI after binding assets and
immutable manifests; retain configs, seeds, environment and prediction receipts
outside the code Git index. For image-source training set `fit.monitor: image.auc`.

## Preserved method behavior

- Xception: native architecture and BatchNorm behavior.
- F3Net: DeepfakeBench's explicitly documented FAD-only profile, not a new claim
  of implementing both branches of the original paper.
- CORE: feature-dependent consistency plus CE, including positional batch halves.
- UCF: paired data, reconstruction and contrastive objectives; not plain CE.
- ProDet: both previously implemented data profiles remain explicit. The new
  benchmark example selects `native_entrypoint`, the executable original
  entry-point's generic image route. The specialized quadruple route still
  requires explicit landmark/neighbor bindings. Its full native-batch CNN
  gradient and full-corpus reproduction gates remain pending.
- Effort: SVD residual parameterization, native augmentation, Adam/CE and existing
  microbatch semantics. VideoAUC remains the selection default; explicit
  `image.auc` now supports image-source validation without inventing video IDs.
- GenD: native CLIP normalization, head choice and norm/LN-name freeze rule.
  The benchmark uses `initialization: pretrained_backbone`: fresh CLIP plus a
  random detection head. The legacy published-detector import remains available
  but is not silently used as the start of fresh-source training.
- ForAda: official ForensicsAdapter reference is pinned separately because it is
  absent from the upstream DeepfakeBench checkout. It retains the query adapter,
  reconstructed CLIP attention, mask decoder and full weighted auxiliary loss.
  See `detectors/forada/README.md` for reference repairs and replay semantics.

## Validation

Use `python tools/validate_benchmark_eight.py --output /absolute/new/run`.
It runs isolated CPU groups serially using the explicit cloud resource backend.
It does not start full dataset training or download checkpoints.
Reference differential checks additionally use the separately pinned
Python3.10/timm0.6.12 environment, via `tools/check_forada_reference.py`.

Cloud containers here have no delegated systemd cgroup. The explicit cloud
runner uses a hard virtual-address-space limit, two-core affinity, inherited
zero-swap cgroup and aggregate RSS/task watchdogs. Its RSS watchdog is not a
kernel hard MemoryMax. Two-worker Effort transport checks request 6GiB RSS
explicitly to retain worker/batch behavior; other groups use 3GiB. No GPU model
job or unbounded fallback is launched. Failures and retries are preserved.
