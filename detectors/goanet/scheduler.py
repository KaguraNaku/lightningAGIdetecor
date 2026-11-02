"""Selected native non-plateau warmup semantics with tensor-safe nested state."""

from torch.optim.lr_scheduler import _LRScheduler


class SourceWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler):
        self.multiplier = 1
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if not self.finished:
                self.after_scheduler.base_lrs = [
                    base * self.multiplier for base in self.base_lrs
                ]
                self.finished = True
            return self.after_scheduler.get_lr()
        return [
            base * (float(self.last_epoch) / self.total_epoch) for base in self.base_lrs
        ]

    def step(self, epoch=None):
        if self.finished:
            self.after_scheduler.step(
                None if epoch is None else epoch - self.total_epoch
            )
        else:
            return super().step(epoch)

    def state_dict(self):
        state = super().state_dict()
        state["after_scheduler"] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        after = state.pop("after_scheduler")
        self.after_scheduler.load_state_dict(after)
        self.__dict__.update(state)
