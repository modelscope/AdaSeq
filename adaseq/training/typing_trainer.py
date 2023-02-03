# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging

from modelscope.preprocessors.base import Preprocessor
from modelscope.trainers.builder import TRAINERS

from adaseq.data.data_collators.base import DataCollatorWithPadding
from adaseq.data.dataset_manager import DatasetManager
from adaseq.metainfo import Trainers

from .default_trainer import DefaultTrainer

logger = logging.getLogger(__name__)


@TRAINERS.register_module(module_name=Trainers.typing_trainer)
class TyingTrainer(DefaultTrainer):
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
            cfg_file=cfg_file,
            work_dir=work_dir,
            dataset_manager=dataset_manager,
            data_collator=data_collator,
            preprocessor=preprocessor,
            seed=seed,
            device=device,
            **kwargs,
        )
        self.model.trainer = self  # add trainer to the model for config
        self.model.expand_vocab()

    def test(self, checkpoint_path=None):
        """Evaluate a trained model on testing set"""
        backup_eval_dataset = self.eval_dataset
        # first evaluate then test
        metric_values = self.evaluate(checkpoint_path)
        # log to terminal
        metric_string = json.dumps(metric_values, indent=2, ensure_ascii=False)
        logger.info('dev: ' + metric_string)

        self.do_test = True
        self.eval_dataset = self.test_dataset
        metric_values = self.evaluate(checkpoint_path)
        self.do_test = False
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
