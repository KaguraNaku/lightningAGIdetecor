"""Selected native two-network initialization, before either optimizer exists."""

from backbones.load_backbone import load_backbone
from detectors.pcl_xception.checkpoint import initialize_backbone

from .detector import SLADDDetector
from .synthesis import SLADDSynthesizer


def create_detector(pretrained):
    backbone = load_backbone("xception_sladd", "vision", model_type="xception_sladd")
    # Main Xception shares the already audited PCL ImageNet import, including
    # fresh classifier/type/magnitude/map heads. The policy has a different
    # partial import and second stem construction; do not use this for it.
    main_origin = initialize_backbone(backbone, pretrained)
    synthesizer = SLADDSynthesizer(pretrained)
    detector = SLADDDetector(backbone, synthesizer)
    detector.initialization = {
        "backbone": main_origin,
        "synthesizer": synthesizer.netG.initialization,
    }
    return detector
