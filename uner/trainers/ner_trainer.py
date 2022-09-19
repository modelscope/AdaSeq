import os
import random
from typing import Callable, Mapping, Optional, Tuple, Union

import torch
from modelscope.metrics import build_metric
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import build_preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.trainers.optimizer.builder import OPTIMIZERS
from modelscope.trainers.parallel.utils import is_parallel
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import ConfigFields, ConfigKeys, ModeKeys
from modelscope.utils.device import create_device, verify_device
from modelscope.utils.logger import get_logger
from modelscope.utils.registry import build_from_cfg, default_group
from modelscope.utils.torch_utils import (
    get_dist_info,
    init_dist,
    set_random_seed,
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
            test_dataset: Optional[Dataset] = None,
            preprocessor: Optional[Preprocessor] = None,
            optimizers: Tuple[torch.optim.Optimizer,
                              torch.optim.lr_scheduler._LRScheduler] = (None,
                                                                        None),
            seed: int = 42,
            **kwargs):

        super(EpochBasedTrainer, self).__init__(cfg_file, arg_parse_fn)

        # add default config
        self.cfg.merge_from_dict(self._get_default_config(), force=False)
        self.cfg = self.rebuild_config(self.cfg)

        # init logger
        self.logger = get_logger(log_level=self.cfg.get('log_level', 'INFO'))

        # seed
        self._seed = 0
        if seed is not None:
            self._seed = seed
        elif has_keys(self.cfg, 'experiment', 'seed'):
            self._seed = self.cfg.experiment.seed
        if self._seed == -1:
            self._seed = random.randint(0, 10000)
        set_random_seed(self._seed)
        self.logger.info('seed: {}'.format(self._seed))

        # work_dir
        if 'work_dir' in kwargs:
            self.work_dir = kwargs['work_dir']
        elif 'work_dir' in self.cfg:
            self.work_dir = self.cfg.work_dir
        elif 'experiment' in self.cfg:
            self.work_dir = os.path.join(str(self.cfg.experiment.exp_dir),
                                         str(self.cfg.experiment.exp_name),
                                         create_datetime_str(), 'outputs')
        else:
            self.work_dir = './work_dir'

        # datasets
        if train_dataset is None and eval_dataset is None:
            if corpus is None:
                corpus = self.build_corpus()
            if corpus.train is not None:
                train_dataset = corpus.train
            if corpus.valid is not None:
                eval_dataset = corpus.valid
            if corpus.test is not None:
                test_dataset = corpus.test

        # labels
        if 'label2id' in kwargs:
            self.label2id = kwargs.pop('label2id')
        elif has_keys(self.cfg, 'dataset', 'label_file'):
            labels = None
            raise NotImplementedError
            self.label2id = dict(zip(labels, range(len(labels))))
        elif train_dataset is not None:
            labels = get_labels(train_dataset)
            self.label2id = dict(zip(labels, range(len(labels))))

        # preprocessor
        self.train_preprocessor, self.eval_preprocessor = None, None
        if isinstance(preprocessor, Preprocessor):
            self.train_preprocessor = preprocessor
            self.eval_preprocessor = preprocessor
        elif isinstance(preprocessor, Mapping):
            if not (ConfigKeys.train in preprocessor
                    or ConfigKeys.val in preprocessor):
                raise ValueError(
                    f'Preprocessor must split with `{ConfigKeys.train}` and `{ConfigKeys.val}` keys!'
                )
            if ConfigKeys.train in preprocessor:
                assert isinstance(preprocessor[ConfigKeys.train], Preprocessor)
                self.train_preprocessor = preprocessor[ConfigKeys.train]
            if ConfigKeys.val in preprocessor:
                assert isinstance(preprocessor[ConfigKeys.val], Preprocessor)
                self.eval_preprocessor = preprocessor[ConfigKeys.val]
        elif hasattr(self.cfg, ConfigFields.preprocessor):
            self.train_preprocessor, self.eval_preprocessor = self.build_preprocessor(
            )

        if self.train_preprocessor is not None:
            self.train_preprocessor.mode = ModeKeys.TRAIN
            self.label2id = self.train_preprocessor.label2id  # update label2id
        if self.eval_preprocessor is not None:
            self.eval_preprocessor.mode = ModeKeys.EVAL

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.logger.info('label2id:', self.label2id)

        # model
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = self.build_model()

        # device
        device_name = kwargs.get('device', 'gpu')
        verify_device(device_name)
        self.device = create_device(device_name)

        # task datasets
        self.train_dataset = self.to_task_dataset(
            train_dataset,
            mode=ModeKeys.TRAIN,
            task_data_config=self.cfg.dataset.get('train', None) if hasattr(
                self.cfg, 'dataset') else None,
            preprocessor=self.train_preprocessor)
        self.eval_dataset = self.to_task_dataset(
            eval_dataset,
            mode=ModeKeys.EVAL,
            task_data_config=self.cfg.dataset.get('valid', None) if hasattr(
                self.cfg, 'dataset') else None,
            preprocessor=self.eval_preprocessor)
        self.test_dataset = self.to_task_dataset(
            test_dataset,
            mode=ModeKeys.EVAL,
            task_data_config=self.cfg.dataset.get('test', None) if hasattr(
                self.cfg, 'dataset') else None,
            preprocessor=self.eval_preprocessor)

        # data collators
        self.train_data_collator, self.eval_default_collate = None, None
        if isinstance(data_collator, Mapping):
            if not (ConfigKeys.train in data_collator
                    or ConfigKeys.val in data_collator):
                raise ValueError(
                    f'data_collator must split with `{ConfigKeys.train}` and `{ConfigKeys.val}` keys!'
                )
            if ConfigKeys.train in data_collator:
                assert isinstance(data_collator[ConfigKeys.train], Callable)
                self.train_data_collator = data_collator[ConfigKeys.train]
            if ConfigKeys.val in data_collator:
                assert isinstance(data_collator[ConfigKeys.val], Callable)
                self.eval_data_collator = data_collator[ConfigKeys.val]
        else:
            default_collate = DataCollatorWithPadding(
                self.train_preprocessor.tokenizer)
            collate_fn = default_collate if data_collator is None else data_collator
            self.train_data_collator = collate_fn
            self.eval_data_collator = collate_fn

        # misc
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

        self._train_iters_per_epoch = kwargs.get('train_iters_per_epoch', None)
        self._eval_iters_per_epoch = kwargs.get('val_iters_per_epoch', None)
        if self._train_iters_per_epoch is None and hasattr(
                self.cfg.train, 'train_iters_per_epoch'):
            self._train_iters_per_epoch = self.cfg.train.train_iters_per_epoch
        if self._eval_iters_per_epoch is None and hasattr(
                self.cfg, 'evaluation') and hasattr(self.cfg.evaluation,
                                                    'val_iters_per_epoch'):
            self._eval_iters_per_epoch = self.cfg.evaluation.val_iters_per_epoch

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

    def build_preprocessor(self) -> Tuple[Preprocessor, Preprocessor]:
        cfg = self.cfg.preprocessor
        if 'model_dir' not in cfg and has_keys(self.cfg, 'model', 'encoder',
                                               'model_dir'):
            cfg['model_dir'] = self.cfg.model.encoder.model_dir
        cfg['label2id'] = self.label2id
        preprocessor = build_preprocessor(cfg)
        train_preprocessor = preprocessor
        eval_preprocessor = preprocessor
        return train_preprocessor, eval_preprocessor

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

    def _init_file_logger(self):
        from modelscope.trainers.hooks.logger.text_logger_hook import TextLoggerHook
        self._file_logger = None
        for hook in self.hooks:
            if isinstance(hook, TextLoggerHook):
               self._file_logger = hook
               break

    def dump_log(self, log_dict):
        if self._file_logger is not None:
            self._file_logger._dump_log(log_dict)

    def test(self, checkpoint_path=None):
        backup_eval_dataset = self.eval_dataset
        self.eval_dataset = self.test_dataset
        metric_values = self.evaluate(checkpoint_path)
        self.eval_dataset = backup_eval_dataset

        # log to terminal
        log_items = []
        for name, val in metric_values.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        self.logger.info('test\t' + ', '.join(log_items))

        # log to file
        self._init_file_logger()
        from collections import OrderedDict
        log_dict = OrderedDict(mode='test', seed=self._seed, **metric_values)
        self.dump_log(log_dict)

        return metric_values
