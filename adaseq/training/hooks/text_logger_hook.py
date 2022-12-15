# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os
import os.path as osp
from collections import OrderedDict

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.logger.text_logger_hook import TextLoggerHook
from modelscope.utils.json_utils import EnhancedEncoder
from modelscope.utils.torch_utils import is_master

logger = logging.getLogger(__name__)


@HOOKS.register_module(module_name=Hooks.TextLoggerHook, force=True)
class AdaSeqTextLoggerHook(TextLoggerHook):
    """Logger hook in text, Output log to both console and local json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedtrainer is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            It is interval of iterations even by_epoch is true. Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        out_dir (str): The directory to save log. If is None, use `trainer.work_dir`
        ignore_rounding_keys (`Union[str, List]`): The keys to ignore float rounding, default 'lr'
        rounding_digits (`int`): The digits of rounding, exceeding parts will be ignored.
    """

    def __init__(
        self,
        by_epoch=True,
        interval=10,
        ignore_last=True,
        reset_flag=False,
        out_dir=None,
        ignore_rounding_keys='lr',
        rounding_digits=5,
        filename: str = 'metrics',
    ) -> None:
        super().__init__(
            by_epoch=by_epoch,
            interval=interval,
            ignore_last=ignore_last,
            reset_flag=reset_flag,
            out_dir=out_dir,
            ignore_rounding_keys=ignore_rounding_keys,
            rounding_digits=rounding_digits,
        )
        self.filename = filename

    def before_run(self, trainer):  # noqa: D102
        super(TextLoggerHook, self).before_run(trainer)

        if self.out_dir is None:
            self.out_dir = trainer.work_dir

        if not osp.exists(self.out_dir) and is_master():
            os.makedirs(self.out_dir)

        self.start_iter = trainer.iter
        self.json_log_path = osp.join(self.out_dir, f'{self.filename}.json')

        logger.info('Text logs will be saved to: %s', self.json_log_path)

    def _dump_log(self, log_dict):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = (
                v if k in self.ignore_rounding_keys else self._round_float(v, self.rounding_digits)
            )

        if is_master():
            with open(self.json_log_path, 'a+') as f:
                json.dump(json_log, f, cls=EnhancedEncoder, ensure_ascii=False)
                f.write('\n')
