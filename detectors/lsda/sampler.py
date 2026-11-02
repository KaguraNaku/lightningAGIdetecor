"""Native group shuffle and one shared slot per five-video LSDA group."""

import random

from torch.utils.data import Sampler


class LSDAGroupSampler(Sampler):
    def __init__(self, num_groups, *, batch_size=10):
        if type(num_groups) is not int or num_groups < 1:
            raise ValueError("LSDA requires a positive group count")
        if type(batch_size) is not int or batch_size < 5 or batch_size % 5:
            raise ValueError("LSDA batch_size must be a positive multiple of five")
        self.num_groups, self.batch_size = num_groups, batch_size
        self.groups_per_batch = batch_size // 5

    def __len__(self):
        return self.num_groups * 5

    def __iter__(self):
        groups = list(range(self.num_groups))
        random.shuffle(groups)
        for start in range(0, len(groups), self.groups_per_batch):
            for group in groups[start : start + self.groups_per_batch]:
                slot = random.randint(0, 31)
                for domain in range(5):
                    yield group * 5 * 32 + domain * 32 + slot
