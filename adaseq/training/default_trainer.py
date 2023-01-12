# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

from datasets.arrow_dataset import Dataset
from modelscope.msdatasets.task_datasets.torch_base_dataset import TorchTaskDataset
from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.builder import build_trainer as ms_build_trainer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import ModeKeys
from torch import nn

from adaseq.data.data_collators.base import DataCollatorWithPadding
from adaseq.data.dataset_manager import DatasetManager
from adaseq.metainfo import Trainers
from adaseq.models.base import Model

from .default_config import DEFAULT_CONFIG
from .lr_scheduler import build_lr_scheduler
from .optimizer import build_optimizer

logger = logging.getLogger(__name__)


@TRAINERS.register_module(module_name=Trainers.default_trainer)
class DefaultTrainer(EpochBasedTrainer):
    """Default trainer class for AdaSeq.

    This trainer inherits from EpochBasedTrainer with some modifications.
    It implements some common data processing functions which are convenient for training a model.
    It also implements a basic test function for evaluate a trained model on the test dataset.

    Args:

    cfg_file (`str`): required
        The path of `Config` of this trial.
    work_dir (`str`): required
        The created directionary to save all produced files in training.
    dataset_manager (`DatasetManager`): required
        A `DatasetManager` with already loaded datasets.
    data_collator (`DataCollatorWithPadding`): required
        A `DataCollatorWithPadding` object to collect data batches.
    preprocessor (`Preprocessor`): required
        A `Preprocessor` to process data before training.
    seed (`int`): required
        Random seed.
    device (`str`): 'cpu' by default
        A device string, e.g., 'cpu', 'cuda', 'cuda:1'.
    kwargs (`dict`):
        Other keyword arguments for `EpochBasedTrainer` of `modelscope`.
    """

    def __init__(
        self,
        cfg_file: str,
        work_dir: str,
        dataset_manager: DatasetManager,
        data_collator: DataCollatorWithPadding,
        preprocessor: Preprocessor,
        seed: int = 42,
        device: str = 'cpu',
        **kwargs,
    ) -> None:
        super().__init__(
            model=None,
            cfg_file=cfg_file,
            cfg_modify_fn=None,
            data_collator=data_collator,
            train_dataset=dataset_manager.train,
            eval_dataset=dataset_manager.valid,
            preprocessor=preprocessor,
            work_dir=work_dir,
            seed=seed,
            device=device,
            **kwargs,
        )

        # Setup testset if there is one
        if dataset_manager.test is not None:
            self.test_dataset = self.to_task_dataset(
                dataset_manager.test,
                mode=ModeKeys.EVAL,
                preprocessor=self.eval_preprocessor,
            )
        logger.info(f'device: {self.device}')

    def rebuild_config(self, config: Config) -> Config:
        """
        Override this func to add adaseq default config.
        """
        config = Config.from_file(config.filename)
        config.merge_from_dict(DEFAULT_CONFIG, force=False)
        return config

    def build_model(self) -> nn.Module:
        """
        Override this func to build adaseq `Model`.
        """
        return Model.from_config(self.cfg)

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        """
        Override this func to customize the optimizer.
        """
        return build_optimizer(self.model, cfg, default_args)

    def build_lr_scheduler(self, cfg: ConfigDict, default_args: dict = None):
        """
        Override this func to customize the lr_scheduler.
        """
        batch_size = self.cfg.train.dataloader.batch_size_per_gpu
        total_steps = max(len(self.train_dataset) / batch_size, 1) * self._max_epochs
        return build_lr_scheduler(cfg, int(total_steps), default_args)

    def to_task_dataset(
        self, dataset: Dataset, mode: str, preprocessor: Preprocessor, **kwargs
    ) -> TorchTaskDataset:
        """
        Override this func to build task dataset from only `datasets.Dataset`.
        """
        task_dataset = TorchTaskDataset(
            dataset, mode=mode, preprocessor=preprocessor, type=self.cfg.model.type
        )
        task_dataset.trainer = self
        return task_dataset

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
        metric_string = json.dumps(metric_values, indent=2, ensure_ascii=False)
        logger.info('test: ' + metric_string)

        # log to file
        self._init_file_logger()
        from collections import OrderedDict

        log_dict = OrderedDict(mode='test', seed=self._seed, **metric_values)
        self.dump_log(log_dict)

        return metric_values


def build_trainer(name: str, **kwargs) -> EpochBasedTrainer:
    """build trainer from config"""
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = kwargs.get('local_rank', '0')
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        kwargs.update(launcher='pytorch', device='gpu')

    trainer = ms_build_trainer(name, kwargs)
    return trainer
