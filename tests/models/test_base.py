import os.path as osp
import shutil
import unittest

import torch
from modelscope.utils.checkpoint import save_configuration
from modelscope.utils.constant import ModelFile

from adaseq.metainfo import Models, Tasks
from adaseq.models.base import Model


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.output_dir = osp.join('tests', 'tmp', 'saved_model')

    def tearDown(self):
        if osp.exists(self.output_dir):
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_save_pretrained(self):
        model = Model.from_config(
            type=Models.sequence_labeling_model,
            task=Tasks.named_entity_recognition,
            id_to_label={0: 'O', 1: 'B', 2: 'I'},
            embedder={'model_name_or_path': 'bert-base-cased'},
        )

        model.save_pretrained(self.output_dir)

        # pytorch_model.bin
        model_file = osp.join(self.output_dir, ModelFile.TORCH_MODEL_BIN_FILE)
        self.assertTrue(osp.isfile(model_file))
        self.assertIn(
            'embedder.transformer_model.embeddings.word_embeddings.weight', torch.load(model_file)
        )

        # configuration.json
        configuration_file = osp.join(self.output_dir, ModelFile.CONFIGURATION)
        self.assertTrue(osp.isfile(configuration_file))

        # config.json
        config_file = osp.join(self.output_dir, 'config.json')
        self.assertTrue(osp.isfile(config_file))


if __name__ == '__main__':
    unittest.main()
