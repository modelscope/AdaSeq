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

    def after_build_dataset(self, **kwargs):
        # get label info from dataset
        if 'label2id' in kwargs:
            self.label2id = kwargs.pop('label2id')
        elif 'train_dataset' in kwargs and kwargs[
                'train_dataset'] is not None and has_keys(
                    self.cfg, 'preprocessor', 'type'):
            labels = get_labels(kwargs.pop('train_dataset'))
            self.label2id = gen_label2id(
                labels, mode=self.cfg.preprocessor.type)
        else:
            raise ValueError('label2id must be set!')
        self.id2label = {v: k for k, v in self.label2id.items()}
        cfg = self.cfg.model
        # num_labels is one of the models super params.
        cfg['num_labels'] = len(self.label2id)

    def build_preprocessor(self,
                           **kwargs) -> Tuple[Preprocessor, Preprocessor]:
        return super().build_preprocessor(label2id=self.label2id, **kwargs)
