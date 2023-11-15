import os.path as osp
import unittest

from modelscope.utils.config import Config

from adaseq.commands.train import build_trainer_from_partial_objects
from tests.models.base import TestModel, compare_fn
from tests.utils import is_huggingface_available


class TestLSTMCRF(TestModel):
    def setUp(self):
        super().setUp()
        cfg_file = osp.join('tests', 'resources', 'configs', 'train_lstm_crf.yaml')
        self.config = Config.from_file(cfg_file)

    @unittest.skipUnless(is_huggingface_available(), 'Cannot connect to huggingface!')
    def test_lstm_crf(self):
        trainer = build_trainer_from_partial_objects(self.config, work_dir=self.tmp_dir, seed=42)

        with self.regress_tool.monitor_ms_train(
            trainer,
            'ut_lstm_crf',
            level='strict',
            compare_fn=compare_fn,
            # Ignore the calculation gap of cpu & gpu
            atol=1e-3,
        ):
            trainer.train()


if __name__ == '__main__':
    unittest.main()
