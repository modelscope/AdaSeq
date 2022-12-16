import os.path as osp
import shutil
import unittest

from adaseq.commands.train import train_model


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.work_dir = osp.join('tests', 'tmp')
        self.cfg_file = osp.join('tests', 'resources', 'configs', 'train_bert_crf.yaml')

    def tearDown(self):
        if osp.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_train_gpu(self):
        train_model(self.cfg_file, work_dir=self.work_dir, force=True, device='gpu')

    def test_train_cpu(self):
        train_model(self.cfg_file, work_dir=self.work_dir, force=True, device='cpu')


if __name__ == '__main__':
    unittest.main()
