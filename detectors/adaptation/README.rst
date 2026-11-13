Explicit UFD adaptation over native detector tasks
=================================================

`UFDAdaptationTask` composes the existing eight Task → Detector → Backbone implementations.
It does not replace their architectures, preprocessing, optimization objectives or trainers.
Fresh fits use the original ImageNet/base-CLIP initialization. Continued fits restore the
source-selected FF++ inference envelope, then use a fresh native optimizer. This is
weight continuation, not exact optimizer/RNG resume.

UCF uses a fixed auxiliary domain vocabulary: genuine0, four FF++ methods1–4, ProGAN5.
Both UFD regimes have six specific outputs. Continuation copies all five old rows and
adds one initialized ProGAN row; the common binary head and all other tensors are retained.
Object categories (airplane, train, etc.) are not mislabeled as different generators.

ForAda's UFD extension derives zero masks for real images and full-image masks for
fully generated ProGAN images. No localized manipulation boundaries are claimed.
Native mask processing and all auxiliary loss coefficients are retained: some boundary
or within-image terms become zero/constant under full-image synthesis, which must be
reported when interpreting results. A shuffled sampler reserves at least one fake per
batch because the native objective is undefined on real-only batches. It visits every
source image once without replacement; native final partial batches remain allowed.

Shared split selection, target RGB exclusion, manifests, budgets, orchestration and
experiment results live in the paper workspace, not this code repository.
