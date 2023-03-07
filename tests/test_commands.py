import os
import shutil
import tempfile
import unittest

from adaseq.commands.train import train_model


class TestCommands(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.cfg_file = os.path.join('tests', 'resources', 'configs', 'train_bert_crf.yaml')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_train_gpu(self):
        train_model(self.cfg_file, work_dir=self.tmp_dir, force=True, device='gpu')

    def test_train_cpu(self):
        train_model(self.cfg_file, work_dir=self.tmp_dir, force=True, device='cpu')


if __name__ == '__main__':
    unittest.main()
