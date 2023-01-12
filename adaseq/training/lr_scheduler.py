# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.trainers.lrscheduler.builder import (
    build_lr_scheduler as ms_build_lr_scheduler,
)
from modelscope.utils.config import Config
from transformers import optimization


def build_lr_scheduler(config: Config, total_steps: int, default_args: Dict[str, Any]):
    """
    Build lr scheduler, `constant` by default.
    """
    if config is None:
        config = dict(type='constant')

    name = config.get('type')
    warmup_rate = config.get('warmup_rate', 0.0)

    # transformers lr_scheduler
    if name in optimization.TYPE_TO_SCHEDULER_FUNCTION:
        return optimization.get_scheduler(
            name,
            default_args['optimizer'],
            num_warmup_steps=int(total_steps * warmup_rate),
            num_training_steps=total_steps,
        )

    # torch lr_scheduler
    else:
        return ms_build_lr_scheduler(config, default_args)
