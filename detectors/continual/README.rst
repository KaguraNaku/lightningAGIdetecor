Selected-state UFD continuation
===============================

``detectors.continual.task:ContinualUFDTask`` composes the existing native UFD
adapter. The first stage uses FF++ source-selected parameters and a fresh UFD
native optimizer. Later stages restore the previous source-selected ``best.pth``
model, native optimizer moments/steps and parameter groups, scaler, objective,
method state, Python/NumPy/Torch/CUDA RNG, both loader generators and sampler
state. Constructing a Python optimizer object to load its state does not perform
a fresh-optimizer update. A changed optimizer type/order/hyperparameter is an
error. Stage-local early stopping resets; optimizer and method clocks do not.

The native training block must be disjoint from all earlier blocks and the fixed
source validation set. UCF retains ordered pairing within the new source block
and its six output domains (real0, FF++1--4, ProGAN5). ForAda keeps its whole-image
targets, native losses and no-replacement fake-guard sampler. Its sampler seed,
batch size and selected-epoch clock are carried onto the new block's indices.
The ProDet native epoch clock is restored as method state.

Only complete epoch/stage boundaries are supported. Persistent data-loader
workers are rejected: workers restart using the carried generator, as in the
frozen protocol. Training and validation generators are separate. Inference
between stages cannot consume the next fit's saved RNG because it is restored
again from the selected training checkpoint. Best weights must never be paired
with final-epoch optimizer state. Actual attempted steps/exposures and the
selected optimization path are reported separately.

The identical successful historical FF++->UFD400 fit is imported instead of
trained again after configuration, source rows and selected model/state checks.
Original predictions, negative later epochs and snapshots remain untouched.
Old checkpoints already contain global RNG and train-loader RNG. The missing
sequential validation-generator state is reconstructed from one base-seed draw
per completed validation iteration, verified against actual zero-worker and
four-spawn-worker torch2.14.1 loaders. ForAda's missing sampler epoch is the
selected completed epoch count. Imports explicitly record these facts; other
PyTorch versions and persistent-worker protocols are refused.

Optional ``fit.gpu_timing`` loads the main-owned, SHA-pinned GPUTimeRecorder.
Real training steps include H2D, forward, backward and native optimizer/scaler;
source prediction calls and checkpoint host time are separate. No synthetic
warmup, extra update or per-batch synchronization is added. Events flush once
at fit completion. Historical imports have null event times and are never
retrained to manufacture timing results. Timing/continuation checks use CPU
fixtures and the unchanged native objectives, not performance benchmarks.
