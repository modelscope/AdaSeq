# Copyright (c) Alibaba, Inc. and its affiliates.
# Fix some modelscope bugs temporarily.
# These bugs will be fixed in the next modelscope version.
import os

from modelscope.trainers.hooks.checkpoint_hook import BestCkptSaverHook, CheckpointHook
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.constant import ModelFile
from modelscope.utils.torch_utils import is_master


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


def suppress_modelscope_ast_warning():  # noqa
    try:
        from modelscope.utils.logger import get_logger

        def filter_modelscope_ast_warning(record):
            return 'not found in ast index file' not in record.msg

        logger = get_logger()
        logger.addFilter(filter_modelscope_ast_warning)
    except IsADirectoryError:
        pass


CheckpointHook._save_pretrained = _save_pretrained
BestCkptSaverHook._should_save = _should_save

suppress_modelscope_ast_warning()
