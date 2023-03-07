import os.path as osp
import unittest

from modelscope.utils.config import Config

from adaseq.commands.train import build_trainer_from_partial_objects
from tests.models.base import TestModel, compare_fn


class TestGlobalPointer(TestModel):
    def setUp(self):
        super().setUp()
        config_dir = osp.join('tests', 'resources', 'configs')
        self.config_span = Config.from_file(osp.join(config_dir, 'train_entity_typing_span.yaml'))
        self.config_concat = Config.from_file(
            osp.join(config_dir, 'train_entity_typing_concat.yaml')
        )

    def test_global_pointer(self):
        trainer = build_trainer_from_partial_objects(
            self.config_span, work_dir=self.tmp_dir, seed=42
        )
        with self.regress_tool.monitor_ms_train(
            trainer,
            'ut_entity_typing',
            level='strict',
            compare_fn=compare_fn,
            # Ignore the calculation gap of cpu & gpu
            atol=1e-3,
        ):
            trainer.train()

    def test_concat_typing(self):
        trainer = build_trainer_from_partial_objects(
            self.config_concat, work_dir=self.tmp_dir, seed=42
        )
        self.is_baseline = True
        with self.regress_tool.monitor_ms_train(
            trainer,
            'ut_concat_entity_typing',
            level='strict',
            compare_fn=compare_fn,
            # Ignore the calculation gap of cpu & gpu
            atol=1e-3,
        ):
            trainer.train()


if __name__ == '__main__':
    unittest.main()
