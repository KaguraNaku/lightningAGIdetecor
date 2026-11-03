ProDet preparation
==================

``ProDetDetector`` composes an injected Backbone with the native1792-to512
projection,three binary experts,feature attention,three stochastic drift blocks,
adjacent-domain Beta mixing andthe final linear binary classifier. The selected
recipe is ``linear_init=False``,``cat_manner=linear``,drift/mixing enabled and
``Op_start=0``. All constructor draws andparameter ordering match the source.

The ordinary ``forward`` returns inference logits. Training calls
``forward_outputs(...,inference=False,epoch=...)`` with the zero-based source
epoch explicitly supplied by the caller. ``loss_terms`` keeps original int64
orFP32 domain labels andmixed confidences. Its three soft cross-entropies,three
drift MSEs andfinal soft cross-entropy retain the source order,in-place overall
alias andgradient paths through the drift targets. The native CPU normal/Beta
draws remain CPU draws;CUDA copies alone are ported to the operand device.

Data recipes require explicit wiring
-----------------------------------

The actual local ``train.py`` does not recognize ``dataset_type=sbiplus_v2``.
Its executed selector chooses the generic base dataset andbatch24 binary images.
The model still divides this batch into four segments;this does not establish
the intended real/blend/BI/deepfake data semantics.

The separately defined ``SBIPlusV2Dataset`` instead collates24 quadruples into
96 images inreal/blend/BI/deepfake order,withoriginal float labels0/1/3/2.
Its original collation was executed ondeclared synthetic marker planes. Actual
SBI/BI synthesis,landmark/nearest-neighbor binding andsource split isolation are
still unported. Keep these recipes distinct;do not flatten specialized domain
labels to binary loss targets orsilently describe a wiring repair as untouched
``train.py`` behavior. The source epoch loop is0..20 inclusive21 epochs.

Verified mathematical scope
---------------------------

* Post-B4 supplied1792x8x8 feature maps atB24->42 andB96->168 match the original
  complete component outputs,all losses,all70 parameter gradients,the input
  feature gradient,one native Adam step andCPU RNG. This excludes the CNN.
* Three full ImageNet-B4 plus randomProDet constructors match782 states,488
  parameter flags/order andCPU RNG. The actual local ImageNet asset is hashed;
  its fresh native unpadded stem andunused randomlast_layer are retained.
* Full32x3x256x256 evaluation matches B4/projection intermediates,all outputs,
  unchanged model states andCPU RNG. The inputs are declared synthetic normalized
  tensors,not evidence of native dataset preprocessing orhistorical scores.
*16 related CPU tests pass,including6 new self-contained component contracts.
  They are separate from the original-source numerical comparisons.

The initial full-constructor harness failed onlist-versus-tuple stride reprs.
The correction canonicalizes only that representation andchecks every numeric
convolution attribute;the source failure andraw checkpoint remain preserved.
Readonly archive reuse occurs only after full typed-value/hash verification.
It does not claim a new ZIP serialization was exercised onthe reuse path.

Pending integration
-------------------

There is no unified ProDet Task,checkpoint importer,source-data adapter,Trainer
orpreset yet. Actual full-CNN gradients at the selected96-image specialized
geometry,fitted/restore/evaluate CLI andthe complete414 CPU suite remain pending.
The last complete386-method pass predates these additions. No full experiment,
historical trained detector orGPU AMP reproduction is claimed.

The full candidate graph's logical unique saved-storage estimates are
7,316,105,712bytes forB24 and29,262,895,872bytes forB96. These are not measured
RAM ordisk budgets. B24 is not substituted for the missing specializedB96 gate.
The two legacy unused ``backbone.last_layer`` parameters stay in the model,
constructor RNG andoptimizer ordering even though features-only training gives
them no gradient. Resource preflights retain8GiB reserves;no batch orprecision
reduction is used to obtain alignment.

Original run IDs,raw tensors,commands,environment,snapshots,failures andresource
receipts are indexed in
``experiments/detector_rewrite/prodet_preparation_audit_20261104.json``.
