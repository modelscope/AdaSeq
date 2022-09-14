from collections.abc import Mapping
import torch


class DataBatch(Mapping):
    keep_fields = ['tokens', 'offset_mapping']
    def __init__(self, batch):
        self.batch = self.tensorize(batch)

    def __repr__(self):
        return str(self.batch)

    def __getitem__(self, key):
        return self.batch.get(key, None)

    def __contains__(self, key):
        return key in self.batch

    def __iter__(self):
        return iter(self.batch)

    def __len__(self):
        return len(self.batch)

    def tensorize(self, batch):
        return {k: torch.tensor(v, dtype=torch.int64 if not k.endswith('mask') else torch.bool)
                if k not in self.keep_fields else v for k, v in batch.items()}

    def to(self, device):
        self.batch = {k: v.to(device) if k not in [self.token_field] else v for k, v in self.batch.items()}

