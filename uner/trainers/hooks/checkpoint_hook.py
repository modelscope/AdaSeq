import os

from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.checkpoint_hook import BestCkptSaverHook
from modelscope.utils.checkpoint import save_checkpoint

from uner.utils.checkpoint import load_checkpoint


@HOOKS.register_module(module_name='MyBestCkptSaverHook')
class MyBestCkptSaverHook(BestCkptSaverHook):
    def before_run(self, trainer):
        super().before_run(trainer)
        self._best_ckpt_file = os.path.join(self.save_dir, 'best.pth')

    def after_run(self, trainer):
        self.logger.info('Loading best checkpoint')
        self._load_checkpoint(trainer, self._best_ckpt_file)
        
    def _save_checkpoint(self, trainer):
        save_checkpoint(trainer.model, self._best_ckpt_file, trainer.optimizer)
        
    def _load_checkpoint(self, trainer, checkpoint_path):
        load_checkpoint(checkpoint_path, trainer.model)

