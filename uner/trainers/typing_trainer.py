# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Tuple

from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS
from torch.utils.data import Dataset

from uner.metainfo import Trainers
from uner.utils.common_utils import has_keys
from .default_trainer import DefaultTrainer


@TRAINERS.register_module(module_name=Trainers.typing_trainer)
class TypingTrainer(DefaultTrainer):
    """Trainer for span typing task."""

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
                            labels.update(span['type'])
            self.labels = sorted(labels)
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
