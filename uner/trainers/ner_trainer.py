import os
import random
from typing import Callable, Optional, Tuple, Union

import torch
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import build_preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.trainers.optimizer.builder import OPTIMIZERS
from modelscope.trainers.parallel.utils import is_parallel
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import build_from_cfg, default_group
from modelscope.utils.torch_utils import (
    create_device,
    get_dist_info,
    init_dist,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from uner.datasets.corpus import Corpus
from uner.metainfo import Trainers
from uner.models.base import Model
from uner.preprocessors.data_collators import DataCollatorWithPadding
from uner.utils.common_utils import create_datetime_str, has_keys
from uner.utils.data_utils import get_labels
from .default_config import DEFAULT_CONFIG


@TRAINERS.register_module(module_name=Trainers.ner_trainer)
class NERTrainer(EpochBasedTrainer):

    def __init__(
            self,
            model: Optional[Union[Model, nn.Module]] = None,
            cfg_file: Optional[str] = None,
            arg_parse_fn: Optional[Callable] = None,
            corpus: Optional[Corpus] = None,
            data_collator: Optional[Callable] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            preprocessor: Optional[Preprocessor] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            **kwargs):

        super(EpochBasedTrainer, self).__init__(cfg_file, arg_parse_fn)
        # add default config
        self.cfg.merge_from_dict(self._get_default_config(), force=False)
        self.cfg = self.rebuild_config(self.cfg)

        # init logger
        self.logger = get_logger(log_level=self.cfg.get('log_level', 'INFO'))

        # seed
        self._seed = 0
        if 'seed' in kwargs and kwargs['seed'] is not None:
            self._seed = kwargs['seed']
        elif has_keys(self.cfg, 'experiment', 'seed'):
            self._seed = self.cfg.experiment.seed

        if self._seed == -1:
            self._seed = random.randint(0, 10000)
        self.logger.info('seed: {}'.format(self._seed))

        # work_dir
        if 'work_dir' in kwargs:
            self.work_dir = kwargs['work_dir']
        else:
            self.work_dir = os.path.join(self.cfg.experiment.exp_dir,
                                         self.cfg.experiment.exp_name,
                                         create_datetime_str())

        if train_dataset is None and eval_dataset is None:
            if corpus is None:
                corpus = self.build_corpus()
            if corpus.train is not None:
                train_dataset = corpus.train
            if corpus.valid is not None:
                eval_dataset = corpus.valid

        if 'label2id' in kwargs:
            self.label2id = kwargs.pop('label2id')
        elif has_keys(self.cfg, 'dataset', 'label_file'):
            labels = None
            raise NotImplementedError
            self.label2id = dict(zip(labels, range(len(labels))))
        elif train_dataset is not None:
            labels = get_labels(train_dataset)
            self.label2id = dict(zip(labels, range(len(labels))))

        self.preprocessor = None
        if isinstance(preprocessor, Preprocessor):
            self.preprocessor = preprocessor
        elif hasattr(self.cfg, 'preprocessor'):
            self.preprocessor = self.build_preprocessor()
        if self.preprocessor is not None:
            self.preprocessor.mode = ModeKeys.TRAIN
            self.label2id = self.preprocessor.label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.logger.info('label2id:', self.label2id)

        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = self.build_model()

        device_name = kwargs.get('device', 'gpu')
        assert device_name in ['gpu',
                               'cpu'], 'device should be either cpu or gpu.'
        self.device = create_device(device_name == 'cpu')

        self.train_dataset = self.to_task_dataset(
            train_dataset, mode=ModeKeys.TRAIN, preprocessor=self.preprocessor)
        self.eval_dataset = self.to_task_dataset(
            eval_dataset, mode=ModeKeys.EVAL, preprocessor=self.preprocessor)

        self.data_collator = data_collator if data_collator is not None \
            else DataCollatorWithPadding(self.preprocessor.tokenizer)

        self.metrics = self.get_metrics()
        self._metric_values = None
        self.optimizers = optimizers
        self._mode = ModeKeys.TRAIN
        self._hooks: List[Hook] = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        if 'max_epochs' not in kwargs:
            assert hasattr(
                self.cfg.train,
                'max_epochs'), 'max_epochs is missing in configuration file'
            self._max_epochs = self.cfg.train.max_epochs
        else:
            self._max_epochs = kwargs['max_epochs']

        self.use_fp16 = kwargs.get('use_fp16', False)

        if kwargs.get('launcher', None) is not None:
            init_dist(kwargs['launcher'])

        self._dist = get_dist_info()[1] > 1

        # model placement
        if self.device.type == 'cuda':
            self.model.to(self.device)
            if not is_parallel(self.model) and self._dist:
                self.model = self.to_parallel(self.model)

    def _get_default_config(self):
        return DEFAULT_CONFIG

    def build_model(self) -> nn.Module:
        cfg = self.cfg.model
        cfg['num_labels'] = len(self.label2id)
        return Model.from_config(cfg)

    def build_corpus(self) -> Corpus:
        corpus = Corpus(task=self.cfg.task, **self.cfg.dataset)
        return corpus

    def build_preprocessor(self) -> Preprocessor:
        cfg = self.cfg.preprocessor
        if 'model_dir' not in cfg and has_keys(self.cfg, 'model', 'encoder',
                                               'model_dir'):
            cfg['model_dir'] = self.cfg.model.encoder.model_dir
        cfg['label2id'] = self.label2id
        return build_preprocessor(cfg)

    def create_optimizer_and_scheduler(self):
        optimizer, lr_scheduler = self.optimizers
        if optimizer is None:
            optimizer_cfg = self.cfg.train.get('optimizer', None)
        else:
            optimizer_cfg = None

        optim_options = {}
        if optimizer_cfg is not None:
            optim_options = optimizer_cfg.pop('options', {})
            optimizer = self.build_optimizer(self.model, cfg=optimizer_cfg)

        if lr_scheduler is None:
            lr_scheduler_cfg = self.cfg.train.get('lr_scheduler', None)
        else:
            lr_scheduler_cfg = None

        lr_options = {}
        if lr_scheduler_cfg is not None:
            assert optimizer is not None
            lr_options = lr_scheduler_cfg.pop('options', {})
            lr_scheduler = build_lr_scheduler(
                cfg=lr_scheduler_cfg, default_args={'optimizer': optimizer})

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        return self.optimizer, self.lr_scheduler, optim_options, lr_options

    @staticmethod
    def build_optimizer(model: nn.Module,
                        cfg: ConfigDict,
                        default_args: dict = None):
        if hasattr(model, 'module'):
            model = model.module
        if default_args is None:
            default_args = {}
        if 'crf_lr' in cfg:
            finetune_parameters = [
                v for k, v in model.named_parameters()
                if v.requires_grad and 'crf' not in k
            ]
            transition_parameters = [
                v for k, v in model.named_parameters()
                if v.requires_grad and 'crf' in k
            ]
            default_args['params'] = [{
                'params': finetune_parameters
            }, {
                'params': transition_parameters,
                'lr': cfg.pop('crf_lr')
            }]
        else:
            default_args['params'] = model.parameters()
        return build_from_cfg(
            cfg,
            OPTIMIZERS,
            group_key=default_group,
            default_args=default_args)
