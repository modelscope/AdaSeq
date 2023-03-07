import os.path as osp
import unittest

from modelscope.utils.config import Config

from adaseq.commands.train import build_trainer_from_partial_objects
from tests.models.base import TestModel, compare_fn


class TestTwoStageNER(TestModel):
    def setUp(self):
        super().setUp()
        cfg_file = osp.join('tests', 'resources', 'configs', 'train_twostage_ner.yaml')
        self.config = Config.from_file(cfg_file)

    def test_two_stage_ner(self):
        trainer = build_trainer_from_partial_objects(self.config, work_dir=self.tmp_dir, seed=42)
        with self.regress_tool.monitor_ms_train(
            trainer,
            'ut_two_stage_ner',
            level='strict',
            compare_fn=compare_fn,
            # Ignore the calculation gap of cpu & gpu
            atol=1e-3,
        ):
            trainer.train()


if __name__ == '__main__':
    unittest.main()
