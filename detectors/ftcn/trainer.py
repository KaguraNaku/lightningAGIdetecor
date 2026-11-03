"""Native whole-batch BCE/source validation; full FTCN backward is unverified."""

from detectors.i3d.trainer import I3DTrainer


class FTCNTrainer(I3DTrainer):
    def fit(self, loader, *, validation_loader, runtime, settings, output_dir):
        # The source loops over start_epoch=0 through nEpochs=10 inclusively.
        # Retain I3D's whole batch8, complete-frame source split validation,
        # probability BCE and retained saved tensors. Audit-only no_grad mmap
        # transport is deliberately absent from this training path.
        return super().fit(
            loader,
            validation_loader=validation_loader,
            runtime=runtime,
            settings={"epochs": 11, **settings},
            output_dir=output_dir,
        )
