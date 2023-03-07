import os
import os.path as osp
import shutil
import tempfile
import unittest

import torch
from modelscope.utils.constant import ModelFile

from adaseq.metainfo import Models, Tasks
from adaseq.models.base import Model


class TestBaseModel(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_save_pretrained(self):
        model = Model.from_config(
            type=Models.sequence_labeling_model,
            task=Tasks.named_entity_recognition,
            id_to_label={0: 'O', 1: 'B', 2: 'I'},
            embedder={'model_name_or_path': 'bert-base-cased'},
        )

        model.save_pretrained(self.tmp_dir)

        # pytorch_model.bin
        model_file = osp.join(self.tmp_dir, ModelFile.TORCH_MODEL_BIN_FILE)
        self.assertTrue(osp.isfile(model_file))
        self.assertIn(
            'embedder.transformer_model.embeddings.word_embeddings.weight', torch.load(model_file)
        )

        # configuration.json
        configuration_file = osp.join(self.tmp_dir, ModelFile.CONFIGURATION)
        self.assertTrue(osp.isfile(configuration_file))

        # config.json
        config_file = osp.join(self.tmp_dir, 'config.json')
        self.assertTrue(osp.isfile(config_file))


if __name__ == '__main__':
    unittest.main()
