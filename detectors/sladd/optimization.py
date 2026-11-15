"""Two disjoint optimizer phases; synthesis update precedes main backward."""

import torch

from .losses import detector_losses, synthesizer_loss


class SLADDOptimization:
    """Method-owned update core; not an epoch loop or an image-synthesis port.

    Call begin_batch before model/synthesizer forward,then step with the native
    generated output dictionary. Complete policy parameters live in synthesizer;
    detector_parameters must match the source filtered main-optimizer order.
    """

    def __init__(
        self,
        detector_parameters,
        synthesizer,
        *,
        learning_rate=2e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=5e-4,
    ):
        main, policy = list(detector_parameters), list(synthesizer.parameters())
        if not main or not policy or {id(p) for p in main} & {id(p) for p in policy}:
            raise ValueError(
                "SLADD requires nonempty disjoint detector/synthesizer parameter sets"
            )
        self.detector_optimizer = torch.optim.Adam(
            main,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        # Native policy fixes beta2=.999 and Adam default eps=1e-8 regardless
        # of main-optimizer settings. Preserve beta1/4 as well as learning_rate/4.
        self.synthesizer_optimizer = torch.optim.Adam(
            policy,
            lr=learning_rate / 4,
            betas=(beta1 / 4, 0.999),
            weight_decay=weight_decay,
        )
        self._begun = False

    def begin_batch(self):
        self.detector_optimizer.zero_grad()
        self._begun = True

    def step(self, outputs):
        if not self._begun:
            raise ValueError("SLADD begin_batch must precede forward and update")
        self._begun = False
        losses = detector_losses(outputs)
        if not torch.isfinite(losses["overall"]):
            raise ValueError("Nonfinite SLADD detector loss; no optimizer update")
        if "map" in outputs and outputs["log_prob"] is not None:
            self.synthesizer_optimizer.zero_grad()
            policy, entropy = synthesizer_loss(
                losses["overall"],
                log_prob=outputs["log_prob"],
                entropy=outputs["entropy"],
            )
            if not torch.isfinite(policy):
                raise ValueError("Nonfinite SLADD policy loss; no policy update")
            if policy.requires_grad:
                policy.backward()
                self._finite_gradients(self.synthesizer_optimizer)
                self.synthesizer_optimizer.step()
        else:
            policy, entropy = 0, 0
        # Keep the independent detector graph intact until after policy.step.
        losses["overall"].backward()
        self._finite_gradients(self.detector_optimizer)
        self.detector_optimizer.step()
        return {**losses, "synthesizer_loss": policy, "entropy_penalty": entropy}

    @staticmethod
    def _finite_gradients(optimizer):
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for group in optimizer.param_groups
            for p in group["params"]
        ):
            raise ValueError("Nonfinite SLADD gradient; optimizer phase stopped")
