import os
import random
from typing import Callable, Mapping, Optional, Tuple, Union

import torch
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import OPTIMIZERS
from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import build_from_cfg, default_group
from torch import nn

from uner.metainfo import Trainers
from uner.models.base import Model
from uner.utils.common_utils import has_keys
from uner.utils.data_utils import gen_label2id, get_labels
from .default_trainer import DefaultTrainer


@TRAINERS.register_module(module_name=Trainers.typing_trainer)
class TypingTrainer(DefaultTrainer):
    """Trainer for span typing task."""

    def after_build_dataset(self, **kwargs):
        # get label info from dataset
        self.labels = None
        self.label2id = None
        if 'label2id' in kwargs:
            self.label2id = kwargs.pop('label2id')
        elif 'train_dataset' in kwargs and kwargs[
                'train_dataset'] is not None and has_keys(
                    self.cfg, 'preprocessor', 'type'):
            self.labels = get_labels(kwargs.pop('train_dataset'))
        else:
            raise ValueError('label2id must be set!')

    def after_build_preprocessor(self, **kwargs):
        if self.train_preprocessor is not None:
            self.label2id = self.train_preprocessor.label2id
        elif self.eval_preprocessor is not None:
            self.label2id = self.eval_preprocessor.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        cfg = self.cfg.model
        # num_labels is one of the models super params.
        cfg['num_labels'] = len(self.label2id)

    def build_preprocessor(self,
                           **kwargs) -> Tuple[Preprocessor, Preprocessor]:
        return super().build_preprocessor(
            labels=self.labels, label2id=self.label2id, **kwargs)
