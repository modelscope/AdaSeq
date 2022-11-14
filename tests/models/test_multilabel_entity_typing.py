import os
import unittest

from modelscope.trainers import build_trainer
from tests.models.base import TestModel, compare_fn

from uner.metainfo import Trainers


class TestGlobalPointer(TestModel):

    def test_global_pointer(self):
        cfg_file = os.path.join('tests', 'resources', 'configs', 'train_entity_typing.yaml')
        trainer = build_trainer(Trainers.typing_trainer, default_args={'cfg_file': cfg_file, 'seed': 42})

        with self.regress_tool.monitor_ms_train(
                trainer,
                'ut_entity_typing',
                level='strict',
                compare_fn=compare_fn,
                # Ignore the calculation gap of cpu & gpu
                atol=1e-3):
            trainer.train()


if __name__ == '__main__':
    unittest.main()
