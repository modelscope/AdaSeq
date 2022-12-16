import os
import unittest

from modelscope.utils.config import Config

from adaseq.commands.train import build_trainer_from_partial_objects
from tests.models.base import TestModel, compare_fn


class TestBiaffineNer(TestModel):
    def test_biaffine_ner(self):
        cfg_file = os.path.join('tests', 'resources', 'configs', 'train_biaffine_ner.yaml')
        config = Config.from_file(cfg_file)
        trainer = build_trainer_from_partial_objects(config, work_dir=config.work_dir, seed=42)

        with self.regress_tool.monitor_ms_train(
            trainer,
            'ut_biaffine_ner',
            level='strict',
            compare_fn=compare_fn,
            # Ignore the calculation gap of cpu & gpu
            atol=1e-3,
        ):
            trainer.train()

        os.remove(config.work_dir + '/config.yaml')


if __name__ == '__main__':
    unittest.main()
