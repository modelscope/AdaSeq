# Copyright (c) Alibaba, Inc. and its affiliates.
# Fix some modelscope bugs temporarily.
# These bugs will be fixed in the next modelscope version.
import os

from modelscope.trainers.hooks.checkpoint_hook import BestCkptSaverHook, CheckpointHook
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.constant import LogKeys, ModelFile
from modelscope.utils.torch_utils import is_master


def _save_checkpoint(self, trainer):
    cur_save_name = self.save_file_name
    if cur_save_name is None:
        if self.by_epoch:
            cur_save_name = os.path.join(
                self.save_dir,
                f'best_{LogKeys.EPOCH}{trainer.epoch + 1}_{self.metric_key}{self._best_metric}.pth',
            )
        else:
            cur_save_name = os.path.join(
                self.save_dir,
                f'best_{LogKeys.ITER}{trainer.iter + 1}_{self.metric_key}{self._best_metric}.pth',
            )
    else:
        if '.' not in cur_save_name:
            cur_save_name = f'{cur_save_name}.pth'
        cur_save_name = os.path.join(self.save_dir, cur_save_name)

    meta = {
        'epoch': trainer.epoch,
        'iter': trainer.iter + 1,
        'inner_iter': trainer.inner_iter + 1,
        'rng_state': self.rng_state,
    }
    for i, hook in enumerate(trainer.hooks):
        meta[f'{hook.__class__}-{i}'] = hook.state_dict()

    if os.path.isfile(cur_save_name):
        os.remove(cur_save_name)
    save_checkpoint(trainer.model, cur_save_name, trainer.optimizer, trainer.lr_scheduler, meta)
    self._best_ckpt_file = cur_save_name
    self._save_pretrained(trainer)


def _save_pretrained(self, trainer):
    output_dir = os.path.join(self.save_dir, ModelFile.TRAIN_OUTPUT_DIR)
    from modelscope.trainers.parallel.utils import is_parallel

    if is_parallel(trainer.model):
        model = trainer.model.module
    else:
        model = trainer.model

    config = trainer.cfg.to_dict()
    # override pipeline by tasks name after finetune done,
    # avoid case like fill mask pipeline with a text cls task
    config['pipeline'] = {'type': config['task']}

    # remove parallel module that is not JSON serializable
    if 'parallel' in config and 'module' in config['parallel']:
        del config['parallel']['module']

    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(
            output_dir,
            ModelFile.TORCH_MODEL_BIN_FILE,
            save_function=save_checkpoint,
            config=config,
            with_meta=False,
        )


def _should_save(self, trainer):
    return is_master() and self._is_best_metric(trainer.metric_values)


def after_run(self, trainer):  # noqa
    if self.restore_best:
        if is_master():
            self.load_checkpoint(self._best_ckpt_file, trainer)


def suppress_modelscope_ast_warning():  # noqa
    from modelscope.utils.logger import get_logger

    def filter_modelscope_ast_warning(record):
        return 'not found in ast index file' not in record.msg

    logger = get_logger('modelscope')
    logger.addFilter(filter_modelscope_ast_warning)


CheckpointHook._save_pretrained = _save_pretrained
BestCkptSaverHook._save_checkpoint = _save_checkpoint
BestCkptSaverHook._should_save = _should_save
BestCkptSaverHook.after_run = after_run

suppress_modelscope_ast_warning()
