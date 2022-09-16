import os

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def load_checkpoint(filename,
                    model,
                    optimizer: Optimizer = None,
                    lr_scheduler: _LRScheduler = None):
    if not os.path.exists(filename):
        raise ValueError(f'Checkpoint file {filename} does not exist!')
    checkpoint = torch.load(filename, map_location='cpu')
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            if isinstance(optimizer, Optimizer):
                optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(optimizer, dict):
                optimizer_dict = checkpoint['optimizer']
                for key, optimizer_ins in optimizer.items():
                    if key in optimizer_dict:
                        optimizer_ins.load_state_dict(optimizer_dict[key])
                    else:
                        logger.warn(
                            f'The state dict of optimizer {key} cannot be found in checkpoint file: {filename}'
                        )
        else:
            logger.warn(
                f'The state dict of optimizer cannot be found in checkpoint file: {filename}'
            )
    if lr_scheduler is not None:
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            logger.warn(
                f'The state dict of lr_scheduler cannot be found in checkpoint file: {filename}'
            )
    state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint[
        'state_dict']
    model.load_state_dict(state_dict)
    if 'meta' in checkpoint:
        return checkpoint.get('meta', {})
