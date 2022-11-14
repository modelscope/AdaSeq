# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Tuple

import torch
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import OPTIMIZERS
from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import build_from_cfg, default_group
from torch import nn
from torch.utils.data import Dataset

from uner.metainfo import Trainers
from uner.models.base import Model
from uner.utils.common_utils import has_keys
from .default_trainer import DefaultTrainer


@TRAINERS.register_module(module_name=Trainers.ner_trainer)
class NERTrainer(DefaultTrainer):
    """ Trainer for NER task."""

    def after_build_dataset(self,
                            train_dataset: Optional[Dataset] = None,
                            eval_dataset: Optional[Dataset] = None,
                            test_dataset: Optional[Dataset] = None,
                            **kwargs):
        # get label info from dataset
        self.labels = None
        self.label2id = None
        if 'label2id' in kwargs:
            self.label2id = kwargs.pop('label2id')
        elif has_keys(self.cfg, 'preprocessor', 'type'):
            labels = set()
            for dataset in (train_dataset, eval_dataset, test_dataset):
                if dataset is not None:
                    for data in dataset:
                        for span in data['spans']:
                            labels.add(span['type'])
            self.labels = sorted(labels)
        else:
            raise ValueError('label2id must be set!')

    def after_build_preprocessor(self, **kwargs):
        # update label2id, since label set was exteded. e.g., B-X->S-X
        if self.train_preprocessor is not None:
            self.label2id = self.train_preprocessor.label2id
        elif self.eval_preprocessor is not None:
            self.label2id = self.eval_preprocessor.label2id

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.logger.info('label2id:', self.label2id)

    def build_preprocessor(self, **kwargs) -> Tuple[Preprocessor, Preprocessor]:
        return super().build_preprocessor(labels=self.labels, label2id=self.label2id, **kwargs)

    def build_model(self) -> nn.Module:
        cfg = self.cfg.model
        # num_labels is one of the models super params.
        cfg['num_labels'] = len(self.label2id)
        return Model.from_config(cfg)

    @staticmethod
    def build_optimizer(model: nn.Module, cfg: ConfigDict, default_args: dict = None):
        if hasattr(model, 'module'):
            model = model.module
        if default_args is None:
            default_args = {}
        if 'crf_lr' in cfg:
            finetune_parameters = [v for k, v in model.named_parameters() if v.requires_grad and 'crf' not in k]
            transition_parameters = [v for k, v in model.named_parameters() if v.requires_grad and 'crf' in k]
            default_args['params'] = [{
                'params': finetune_parameters
            }, {
                'params': transition_parameters,
                'lr': cfg.pop('crf_lr')
            }]
        else:
            default_args['params'] = model.parameters()
        return build_from_cfg(cfg, OPTIMIZERS, group_key=default_group, default_args=default_args)
