# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Tuple

import transformers
from datasets import DownloadManager
from datasets.utils.file_utils import is_remote_url
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.trainers.optimizer.builder import OPTIMIZERS
from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import build_from_cfg, default_group
from torch import nn
from torch.utils.data import Dataset

from adaseq.metainfo import Trainers
from adaseq.utils.common_utils import has_keys
from .default_trainer import DefaultTrainer


@TRAINERS.register_module(module_name=Trainers.typing_trainer)
class TypingTrainer(DefaultTrainer):
    """Trainer for span typing task."""

    def after_build_dataset(self,
                            train_dataset: Optional[Dataset] = None,
                            eval_dataset: Optional[Dataset] = None,
                            test_dataset: Optional[Dataset] = None,
                            **kwargs):
        """ Collect labels from train/eval/test datasets and create label to id mapping """
        # get label info from dataset
        self.labels = None
        self.label2id = None
        dataset_cfg = self.cfg.dataset
        if 'label_file' in dataset_cfg:  # TODO: refactor
            label_file = dataset_cfg.pop('label_file')
            if is_remote_url(label_file):
                label_file = DownloadManager('tmp_typing').download(label_file)
            self.labels = sorted(line.strip() for line in open(label_file))
        elif 'label2id' in kwargs:
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
        """ Update label2id, since label set was extended. e.g., B-X->S-X """
        if self.train_preprocessor is not None:
            self.label2id = self.train_preprocessor.label2id
        elif self.eval_preprocessor is not None:
            self.label2id = self.eval_preprocessor.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        cfg = self.cfg.model
        # num_labels is one of the models super params.
        cfg['num_labels'] = len(self.label2id)
        cfg['labels'] = self.labels

    def build_preprocessor(self, **kwargs) -> Tuple[Preprocessor, Preprocessor]:
        """ Build preprocessor with labels and label2id """
        return super().build_preprocessor(labels=self.labels, label2id=self.label2id, **kwargs)

    @staticmethod
    def build_optimizer(model: nn.Module, cfg: ConfigDict, default_args: dict = None):
        """ Builde layer-wise lr optimizer """

        if hasattr(model, 'module'):
            model = model.module
        if default_args is None:
            default_args = {}
        if 'decoder_lr' in cfg:
            finetune_parameters = [v for k, v in model.named_parameters() if v.requires_grad and 'decoder' not in k]
            decoder_parameters = [v for k, v in model.named_parameters() if v.requires_grad and 'decoder' in k]
            default_args['params'] = [{
                'params': finetune_parameters
            }, {
                'params': decoder_parameters,
                'lr': cfg.pop('decoder_lr')
            }]
        else:
            default_args['params'] = model.parameters()
        return build_from_cfg(cfg, OPTIMIZERS, group_key=default_group, default_args=default_args)

    def create_optimizer_and_scheduler(self):
        """ Create optimizer and lr-scheduler from config,
        support huggingface CosineLR for typing """
        optimizer, lr_scheduler = self.optimizers
        if optimizer is None:
            optimizer_cfg = self.cfg.train.get('optimizer', None)
        else:
            optimizer_cfg = None

        optim_options = {}
        if optimizer_cfg is not None:
            optim_options = optimizer_cfg.pop('options', {})
            optimizer = self.build_optimizer(self.model, cfg=optimizer_cfg)  # support customize optimizer

        if lr_scheduler is None:
            lr_scheduler_cfg = self.cfg.train.get('lr_scheduler', None)
        else:
            lr_scheduler_cfg = None

        lr_options = {}
        if lr_scheduler_cfg is not None:
            lr_options = lr_scheduler_cfg.pop('options', {})
            assert optimizer is not None
            if lr_scheduler_cfg.get('type', '') == 'CosineLR':
                epoch = self.cfg.train.get('max_epochs', None)
                assert epoch is not None
                dataloader_cfg = self.cfg.train.get('dataloader', None)
                assert dataloader_cfg is not None
                bz = dataloader_cfg.get('batch_size_per_gpu', None)
                assert bz is not None
                iters = len(self.train_dataset) // bz * epoch
                warmup_rate = lr_scheduler_cfg.get('warmup_rate', 0.0)
                lr_scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=int(iters * warmup_rate), num_training_steps=int(iters))
            else:
                lr_scheduler = build_lr_scheduler(cfg=lr_scheduler_cfg, default_args={'optimizer': optimizer})

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        return self.optimizer, self.lr_scheduler, optim_options, lr_options
