import os
import shutil
import tempfile
import unittest
from unittest import mock

from modelscope.pipelines import pipeline
from modelscope.utils.constant import ModelFile, Tasks

from adaseq import pipelines
from adaseq.commands.train import train_model
from tests.utils import is_huggingface_available


class TestPipelines(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        try:
            from modelscope.models.base import base_model
            from modelscope.pipelines import builder

            builder.register_plugins_repo = mock.Mock(return_value=None)
            base_model.register_plugins_repo = mock.Mock(return_value=None)
        except (ImportError, AttributeError):
            pass

        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_train_then_sequence_labeling_pipeline(self):
        cfg_file = os.path.join(
            'tests', 'resources', 'configs', 'finetune_bert_crf_from_trained_model.yaml'
        )
        train_model(cfg_file, work_dir=self.tmp_dir, force=True)

        pipeline_ins = pipeline(
            Tasks.named_entity_recognition,
            os.path.join(self.tmp_dir, ModelFile.TRAIN_BEST_OUTPUT_DIR),
        )
        print(pipeline_ins('据法国统计和经济研究中心１９９６年调查，６０岁以上的人已占法国人口总数的２３％，而１０年前还仅为１８．４％。'))

    @unittest.skipUnless(is_huggingface_available(), 'Cannot connect to huggingface!')
    def test_span_based_ner_pipeline(self):
        pipeline_ins = pipeline(
            Tasks.named_entity_recognition,
            'damo/nlp_nested-ner_named-entity-recognition_chinese-base-med',
        )
        print(
            pipeline_ins(
                '1、可测量目标： 1周内胸闷缓解。2、下一步诊疗措施：1.心内科护理常规，一级护理，低盐低脂饮食，留陪客。2.予“阿司匹林肠溶片”抗血小板聚集，“呋塞米、螺内酯”利尿减轻心前负荷，“瑞舒伐他汀”调脂稳定斑块，“厄贝沙坦片片”降血压抗心机重构'
            )
        )

    @unittest.skipUnless(is_huggingface_available(), 'Cannot connect to huggingface!')
    def test_maoe_pipelines(self):
        pipeline_ins = pipeline(
            Tasks.named_entity_recognition,
            'damo/nlp_maoe_named-entity-recognition_chinese-base-general',
            model_revision='v0.0.1',
        )
        print(pipeline_ins('刘培强，男，生理年龄40岁（因为在太空中进入休眠状态），实际年龄52岁，领航员国际空间站中的中国航天员，机械工程专家，军人，军衔中校。'))


if __name__ == '__main__':
    unittest.main()
