import os
import unittest

from modelscope.models.nlp import SbertModel
from modelscope.models.nlp.bert.backbone import BertModel as MsBertModel
from modelscope.models.nlp.structbert.backbone import SbertModel
from transformers import BertModel

from adaseq.metainfo import Models, Tasks
from adaseq.models.base import Model
from adaseq.models.sequence_labeling_model import SequenceLabelingModel
from adaseq.modules.embedders import Embedder
from tests.utils import is_huggingface_available


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.bert_config_file = os.path.join('tests', 'resources', 'configs', 'bert.yaml')
        self.structbert_config_file = os.path.join(
            'tests', 'resources', 'configs', 'structbert.yaml'
        )

    @unittest.skipUnless(is_huggingface_available(), 'Cannot connect to huggingface!')
    def test_get_embedder_from_huggingface(self):
        """可以不使用名字从huggingface初始化一个embedder"""
        model = Embedder.from_config(model_name_or_path='bert-base-cased')
        self.assertTrue(isinstance(model.transformer_model, BertModel))

    def test_get_embedder_from_modelscope(self):
        """可以使用名字从modelscope初始化一个embedder"""
        model = Embedder.from_config(model_name_or_path='damo/nlp_structbert_backbone_base_std')
        self.assertTrue(isinstance(model.transformer_model, SbertModel))

    def test_get_embedder_from_modelscope_task_model(self):
        """可以使用名字从modelscope任务模型初始化一个embedder"""
        model = Embedder.from_config(
            model_name_or_path='damo/nlp_raner_named-entity-recognition_chinese-base-news'
        )
        self.assertTrue(isinstance(model.transformer_model, (BertModel, MsBertModel)))

    @unittest.skipUnless(is_huggingface_available(), 'Cannot connect to huggingface!')
    def test_get_embedder_from_cfg_bert(self):
        """可以指定配置文件初始化一个embedder，本例中配置文件中存在model_name_or_path参数"""
        model = Embedder.from_config(cfg_dict_or_path=self.bert_config_file)
        self.assertTrue(isinstance(model.transformer_model, BertModel))

    def test_get_embedder_from_cfg_structbert(self):
        """可以指定配置文件初始化一个embedder，本例中配置文件中存在model_name_or_path参数"""
        model = Embedder.from_config(cfg_dict_or_path=self.structbert_config_file)
        self.assertTrue(isinstance(model.transformer_model, SbertModel))

    @unittest.skipUnless(is_huggingface_available(), 'Cannot connect to huggingface!')
    def test_get_model_from_huggingface(self):
        """可以不配置文件初始化一个模型"""
        model = Model.from_config(
            type=Models.sequence_labeling_model,
            task=Tasks.named_entity_recognition,
            id_to_label={0: 'O', 1: 'B'},
            embedder={'model_name_or_path': 'bert-base-cased'},
        )
        self.assertTrue(isinstance(model, SequenceLabelingModel))

    def test_get_model_from_modelscope(self):
        """可以不配置文件初始化一个模型"""
        model = Model.from_config(
            type=Models.sequence_labeling_model,
            task=Tasks.named_entity_recognition,
            id_to_label={0: 'O', 1: 'B'},
            embedder={'model_name_or_path': 'damo/nlp_structbert_backbone_base_std'},
        )
        self.assertTrue(isinstance(model, SequenceLabelingModel))

    @unittest.skipUnless(is_huggingface_available(), 'Cannot connect to huggingface!')
    def test_get_model_from_cfg_bert(self):
        """可以指定配置文件初始化一个模型，本例中配置文件中存在model_name_or_path参数"""
        model = Model.from_config(cfg_dict_or_path=self.bert_config_file)
        self.assertTrue(isinstance(model, SequenceLabelingModel))
        self.assertTrue(isinstance(model.embedder.transformer_model, BertModel))

    def test_get_model_from_cfg_structbert(self):
        """可以指定配置文件初始化一个模型，本例中配置文件中存在model_name_or_path参数"""
        model = Model.from_config(cfg_dict_or_path=self.structbert_config_file)
        self.assertTrue(isinstance(model, SequenceLabelingModel))
        self.assertTrue(isinstance(model.embedder.transformer_model, SbertModel))


if __name__ == '__main__':
    unittest.main()
