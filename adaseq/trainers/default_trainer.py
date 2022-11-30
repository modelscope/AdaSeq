# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
from typing import Callable, List, Mapping, Optional, Tuple, Union

from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import build_preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks import Hook
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.parallel.utils import is_parallel
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import ConfigFields, ConfigKeys, ModeKeys
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import (
    get_dist_info,
    get_local_rank,
    init_dist,
    set_random_seed,
)
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorMixin

from adaseq.data.data_collators.base import build_data_collator
from adaseq.data.dataset_manager import DatasetManager
from adaseq.metainfo import Preprocessors, Trainers
from adaseq.models.base import Model
from adaseq.utils.common_utils import create_datetime_str, has_keys

from .default_config import DEFAULT_CONFIG

BUILTIN_PREPROCESSOR = set(
    getattr(Preprocessors, _a) for _a in dir(Preprocessors) if not _a.startswith('__')
)


@TRAINERS.register_module(module_name=Trainers.default_trainer)
class DefaultTrainer(EpochBasedTrainer):
    """Default trainer class for AdaSeq.

    This trainer inherits from EpochBasedTrainer with some modifications.
    It implements some common data processing functions which are convenient for training a model.
    It also implements a basic test function for evaluate a trained model on the test dataset.
    """

    def __init__(
        self,
        model: Optional[Union[Model, nn.Module]] = None,
        cfg_file: Optional[str] = None,
        arg_parse_fn: Optional[Callable] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        preprocessor: Optional[Preprocessor] = None,
        optimizers: Tuple[Optimizer, LRScheduler] = (
            None,
            None,
        ),
        seed: int = 42,
        **kwargs,
    ):

        super(EpochBasedTrainer, self).__init__(cfg_file, arg_parse_fn)

        # add default config
        self.cfg.merge_from_dict(self._get_default_config(), force=False)
        self.cfg = self.rebuild_config(self.cfg)
        self.meta = self.cfg.to_dict()

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
            self.work_dir = os.path.join(
                str(self.cfg.experiment.exp_dir),
                str(self.cfg.experiment.exp_name),
                'outputs',
                create_datetime_str(),
            )
        else:
            self.work_dir = './work_dir'

        # datasets
        if train_dataset is None and eval_dataset is None and test_dataset is None:
            dataset = self.build_dataset()
            if dataset.train is not None:
                train_dataset = dataset.train
            if dataset.valid is not None:
                eval_dataset = dataset.valid
            if dataset.test is not None:
                test_dataset = dataset.test

        self.after_build_dataset(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        # preprocessor
        self.train_preprocessor, self.eval_preprocessor = None, None
        if isinstance(preprocessor, Preprocessor):
            self.train_preprocessor = preprocessor
            self.eval_preprocessor = preprocessor
        elif isinstance(preprocessor, Mapping):
            if not (ConfigKeys.train in preprocessor or ConfigKeys.val in preprocessor):
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
            self.train_preprocessor, self.eval_preprocessor = self.build_preprocessor()

        if self.eval_preprocessor is not None:
            self.eval_preprocessor.mode = ModeKeys.EVAL
            self.tokenizer = self.eval_preprocessor.tokenizer
        if self.train_preprocessor is not None:
            self.train_preprocessor.mode = ModeKeys.TRAIN
            self.tokenizer = self.train_preprocessor.tokenizer

        self.after_build_preprocessor(**kwargs)

        # model
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = self.build_model()

        # device
        if kwargs.get('launcher', None) is not None:
            init_dist(kwargs['launcher'])

        _, world_size = get_dist_info()
        self._dist = world_size > 1

        device_name = kwargs.get('device', 'gpu')
        if self._dist:
            local_rank = get_local_rank()
            device_name = f'cuda:{local_rank}'

        self.device = create_device(device_name)

        # task datasets
        self.train_dataset = self.to_task_dataset(
            train_dataset,
            mode=ModeKeys.TRAIN,
            task_data_config=self.cfg.dataset.get('train', None)
            if hasattr(self.cfg, 'dataset')
            else None,
            preprocessor=self.train_preprocessor,
        )
        self.eval_dataset = self.to_task_dataset(
            eval_dataset,
            mode=ModeKeys.EVAL,
            task_data_config=self.cfg.dataset.get('valid', None)
            if hasattr(self.cfg, 'dataset')
            else None,
            preprocessor=self.eval_preprocessor,
        )
        self.test_dataset = self.to_task_dataset(
            test_dataset,
            mode=ModeKeys.EVAL,
            task_data_config=self.cfg.dataset.get('test', None)
            if hasattr(self.cfg, 'dataset')
            else None,
            preprocessor=self.eval_preprocessor,
        )

        # data collators
        self.train_data_collator, self.eval_data_collate = None, None
        if isinstance(data_collator, Mapping):
            if not (ConfigKeys.train in data_collator or ConfigKeys.val in data_collator):
                raise ValueError(
                    f'data_collator must split with `{ConfigKeys.train}` and `{ConfigKeys.val}` keys!'
                )
            if ConfigKeys.train in data_collator:
                assert isinstance(data_collator[ConfigKeys.train], Callable)
                self.train_data_collator = data_collator[ConfigKeys.train]
            if ConfigKeys.val in data_collator:
                assert isinstance(data_collator[ConfigKeys.val], Callable)
                self.eval_data_collator = data_collator[ConfigKeys.val]
        elif data_collator is not None:
            self.train_data_collator = data_collator
            self.eval_data_collator = data_collator
        else:
            self.train_data_collator, self.eval_data_collator = self.build_data_collator()

        # misc
        self.metrics = self.get_metrics()
        self._metric_values = None
        self.optimizer = None
        self.lr_scheduler = None
        self.optimizers = optimizers
        self._mode = ModeKeys.TRAIN
        self._hooks: List[Hook] = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        if 'max_epochs' not in kwargs:
            assert hasattr(
                self.cfg.train, 'max_epochs'
            ), 'max_epochs is missing in configuration file'
            self._max_epochs = self.cfg.train.max_epochs
        else:
            self._max_epochs = kwargs['max_epochs']

        self._train_iters_per_epoch = kwargs.get('train_iters_per_epoch', None)
        self._eval_iters_per_epoch = kwargs.get('val_iters_per_epoch', None)
        if self._train_iters_per_epoch is None and hasattr(self.cfg.train, 'train_iters_per_epoch'):
            self._train_iters_per_epoch = self.cfg.train.train_iters_per_epoch
        if (
            self._eval_iters_per_epoch is None
            and hasattr(self.cfg, 'evaluation')
            and hasattr(self.cfg.evaluation, 'val_iters_per_epoch')
        ):
            self._eval_iters_per_epoch = self.cfg.evaluation.val_iters_per_epoch

        self.use_fp16 = kwargs.get('use_fp16', False)

        # model placement
        if self.device.type == 'cuda':
            self.model.to(self.device)
            if not is_parallel(self.model) and self._dist:
                self.model = self.to_parallel(self.model)

    def _get_default_config(self):
        return DEFAULT_CONFIG

    def build_model(self) -> nn.Module:
        """Build model from config"""
        return Model.from_config(self.cfg)

    def build_dataset(self) -> DatasetManager:
        """Build dataset from config"""
        dataset = DatasetManager(task=self.cfg.task, **self.cfg.dataset)
        return dataset

    def after_build_dataset(self, **kwargs):
        """Do something after building dataset"""
        pass

    def build_preprocessor(self, **kwargs) -> Tuple[Preprocessor, Preprocessor]:
        """Build preprocessor from config"""
        cfg = self.cfg.preprocessor
        if 'model_dir' not in cfg and has_keys(self.cfg, 'model', 'encoder', 'model_name_or_path'):
            cfg['model_dir'] = self.cfg.model.encoder.model_name_or_path
        for k, v in kwargs.items():
            cfg[k] = v

        field_name = cfg.pop('field_name', None)
        if field_name is None and cfg['type'] in BUILTIN_PREPROCESSOR:
            field_name = None
        else:
            field_name = field_name or 'nlp'

        preprocessor = build_preprocessor(cfg, field_name)  # type: ignore
        train_preprocessor = preprocessor
        eval_preprocessor = preprocessor
        return train_preprocessor, eval_preprocessor

    def after_build_preprocessor(self, **kwargs):
        """Do something after building preprocessor"""
        pass

    def build_data_collator(self) -> Tuple[DataCollatorMixin, DataCollatorMixin]:
        """Build data collator from config"""
        cfg = self.cfg.data_collator
        if isinstance(cfg, str):
            cfg = dict(type=cfg)
        data_collator = build_data_collator(self.tokenizer, cfg)
        train_data_collator = data_collator
        eval_data_collator = data_collator
        return train_data_collator, eval_data_collator

    def create_optimizer_and_scheduler(self):
        """Create optimizer and lr-scheduler from config"""
        optimizer, lr_scheduler = self.optimizers
        if optimizer is None:
            optimizer_cfg = self.cfg.train.get('optimizer', None)
        else:
            optimizer_cfg = None

        optim_options = {}
        if optimizer_cfg is not None:
            optim_options = optimizer_cfg.pop('options', {})
            optimizer = self.build_optimizer(
                self.model, cfg=optimizer_cfg
            )  # support customize optimizer

        if lr_scheduler is None:
            lr_scheduler_cfg = self.cfg.train.get('lr_scheduler', None)
        else:
            lr_scheduler_cfg = None

        lr_options = {}
        if lr_scheduler_cfg is not None:
            assert optimizer is not None
            lr_options = lr_scheduler_cfg.pop('options', {})
            lr_scheduler = build_lr_scheduler(
                cfg=lr_scheduler_cfg, default_args={'optimizer': optimizer}
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        return self.optimizer, self.lr_scheduler, optim_options, lr_options

    @staticmethod
    def build_optimizer(model: nn.Module, cfg: ConfigDict, default_args: dict = None):
        """Build optimizer from config"""
        return build_optimizer(model, cfg, default_args)

    def _init_file_logger(self):
        from modelscope.trainers.hooks.logger.text_logger_hook import TextLoggerHook

        self._file_logger = None
        for hook in self.hooks:
            if isinstance(hook, TextLoggerHook):
                self._file_logger = hook
                break

    def dump_log(self, log_dict):
        """Dump dict to log file"""
        if self._file_logger is not None:
            self._file_logger._dump_log(log_dict)

    def train(self, checkpoint_path=None, *args, **kwargs):
        """Train a model with training set"""
        return super().train(checkpoint_path=checkpoint_path, *args, **kwargs)

    def test(self, checkpoint_path=None):
        """Evaluate a trained model on testing set"""
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
