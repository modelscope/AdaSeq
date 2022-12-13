import os
import unittest

from modelscope.utils.config import Config

from adaseq.metainfo import Trainers
from adaseq.training import build_trainer
from tests.models.base import TestModel, compare_fn


class TestGlobalPointer(TestModel):
    def test_global_pointer(self):
        cfg_file = os.path.join('tests', 'resources', 'configs', 'train_global_pointer.yaml')
        config = Config.from_file(cfg_file)
        trainer = build_trainer(Trainers.default_trainer, config, work_dir=config.work_dir, seed=42)

        with self.regress_tool.monitor_ms_train(
            trainer,
            'ut_global_pointer',
            level='strict',
            compare_fn=compare_fn,
            # Ignore the calculation gap of cpu & gpu
            atol=1e-3,
        ):
            trainer.train()

        os.remove(config.work_dir + '/config.yaml')


if __name__ == '__main__':
    unittest.main()
