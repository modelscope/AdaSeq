import os.path as osp
import unittest

from modelscope.utils.config import Config

from adaseq.commands.train import build_trainer_from_partial_objects
from adaseq.exporters.sequence_labeling_model_exporter import (
    SequenceLabelingModelExporter,
)
from tests.models.base import TestModel, compare_fn


class TestBertCRF(TestModel):
    def setUp(self):
        super().setUp()
        cfg_file = osp.join('tests', 'resources', 'configs', 'train_bert_crf.yaml')
        self.config = Config.from_file(cfg_file)

    def test_train_bert_crf(self):
        trainer = build_trainer_from_partial_objects(self.config, work_dir=self.tmp_dir, seed=42)

        with self.regress_tool.monitor_ms_train(
            trainer,
            'ut_bert_crf',
            level='strict',
            compare_fn=compare_fn,
            # Ignore the calculation gap of cpu & gpu
            atol=1e-3,
        ):
            trainer.train()

    def test_export_bert_crf(self):
        trainer = build_trainer_from_partial_objects(self.config, work_dir=self.tmp_dir)
        model = trainer.model
        preprocessor = trainer.eval_preprocessor
        exporter = SequenceLabelingModelExporter(model=model, preprocessor=preprocessor)
        with self.subTest(format='onnx'):
            print(exporter.export_onnx(output_dir=self.tmp_dir))
        with self.subTest(format='torchscript'):
            print(exporter.export_torch_script(output_dir=self.tmp_dir, strict=False))


if __name__ == '__main__':
    unittest.main()
