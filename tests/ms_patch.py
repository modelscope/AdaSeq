from collections.abc import Mapping

import numpy as np
from torch import distributed as dist


def train_step(self, model, inputs):
    # EvaluationHook will do evaluate and change mode to val, return to train mode
    # TODO: find more pretty way to change mode
    model.train()
    self._mode = 'train'
    # call model forward but not __call__ to skip postprocess
    train_outputs = model.forward(inputs)

    if not isinstance(train_outputs, dict):
        raise TypeError('"model.forward()" must return a dict')

    # add model output info to log
    if 'log_vars' not in train_outputs:
        default_keys_pattern = ['loss']
        match_keys = set([])
        for key_p in default_keys_pattern:
            match_keys.update([key for key in train_outputs.keys() if key_p in key])

        log_vars = {}
        for key in match_keys:
            value = train_outputs.get(key, None)
            if value is not None:
                if dist.is_available() and dist.is_initialized():
                    value = value.data.clone().to('cuda')
                    dist.all_reduce(value.div_(dist.get_world_size()))
                log_vars.update({key: value.item()})
        self.log_buffer.update(log_vars)
    else:
        self.log_buffer.update(train_outputs['log_vars'])

    self.train_outputs = train_outputs


def numpify_tensor_nested(tensors, reduction=None, clip_value=10000):
    import torch

    # if isinstance(tensors, torch.Tensor) and len(tensors.shape) > 0 and tensors.shape[0] == 16:
    #     print()
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(numpify_tensor_nested(t, reduction, clip_value) for t in tensors)
    if isinstance(tensors, Mapping):
        return {k: numpify_tensor_nested(t, reduction, clip_value) for k, t in tensors.items()}
    if isinstance(tensors, torch.Tensor):
        t: np.ndarray = tensors.cpu().numpy()
        if clip_value is not None:
            t = np.where(t > clip_value, clip_value, t)
            t = np.where(t < -clip_value, -clip_value, t)
        if reduction == 'sum':
            return t.sum(dtype=np.float)
        elif reduction == 'mean':
            return t.mean(dtype=np.float)
        return t
    return tensors
