import os
import unittest

from adaseq.data.dataset_manager import DatasetManager


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.setUp_ner_dataset()

    def setUp_ner_dataset(self):
        self.ner_task = 'named-entity-recognition'
        self.ner_local_dir = os.path.abspath(
            os.path.join('tests', 'resources', 'datasets', 'toy_msra')
        )
        self.ner_local_python_script = os.path.abspath(
            os.path.join('tests', 'resources', 'datasets', 'toy_msra', 'toy_msra.py')
        )
        self.ner_local_zip_file = os.path.abspath(
            os.path.join('tests', 'resources', 'datasets', 'toy_msra', 'toy_msra.zip')
        )
        self.ner_local_train_file = os.path.abspath(
            os.path.join('tests', 'resources', 'datasets', 'toy_msra', 'train.txt')
        )
        self.ner_local_test_file = os.path.abspath(
            os.path.join('tests', 'resources', 'datasets', 'toy_msra', 'test.txt')
        )
        self.ner_remote_corpus_id = 'toy_msra'
        url_prefix = 'https://www.modelscope.cn/api/v1/datasets/izhx404/toy_msra/repo/files?Revision=master&FilePath='
        self.ner_remote_file = url_prefix + 'toy_msra.zip'
        self.ner_remote_splits = {
            'train': url_prefix + 'train.txt',
            'test': url_prefix + 'test.txt',
        }
        self.ner_train_data_size = 1000
        self.ner_test_data_size = 200
        self.ner_expected_data = {
            'id': '40',
            'tokens': [
                '国',
                '正',
                '先',
                '生',
                '在',
                '我',
                '心',
                '中',
                '就',
                '是',
                '这',
                '样',
                '的',
                '一',
                '位',
                '学',
                '长',
                '。',
            ],
            'spans': [{'start': 0, 'end': 2, 'type': 'PER'}],
            'mask': [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        }

    def _select_and_test(self, dm: DatasetManager, mask: bool = False, field_strict=True):
        if mask:
            ner_expected_data = self.ner_expected_data
        else:
            ner_expected_data = self.ner_expected_data.copy()
            ner_expected_data.pop('mask')

        selected_data = dm.train[int(ner_expected_data['id'])]
        if not field_strict:
            # dataset like conll2003 from huggingface may have other fields
            selected_data = {k: selected_data[k] for k in ner_expected_data}
        self.assertEqual(len(dm.train), self.ner_train_data_size)
        self.assertEqual(len(dm.test), self.ner_test_data_size)
        self.assertEqual(selected_data, ner_expected_data)

    def test_load_local_named_entity_recognition_dataset_files(self):
        """可以指定train文件初始化一个Corpus"""
        dm = DatasetManager.from_config(
            task=self.ner_task,
            data_file=dict(train=self.ner_local_train_file, test=self.ner_local_test_file),
            data_type='conll',
        )
        self._select_and_test(dm, True)

    def test_load_remote_named_entity_recognition_dataset_files(self):
        """可以指定远程 train 文件初始化一个Corpus"""
        dm = DatasetManager.from_config(
            task=self.ner_task,
            data_file=self.ner_remote_splits,
            data_type='conll',
        )
        self._select_and_test(dm, True)

    def test_load_local_named_entity_recognition_dataset_zip(self):
        """可以指定 dataset文件 从本地初始化一个Corpus"""
        dm = DatasetManager.from_config(
            task=self.ner_task,
            data_file=self.ner_local_zip_file,
            data_type='conll',
        )
        self._select_and_test(dm, True)

    def test_load_local_named_entity_recognition_dataset_script(self):
        """可以指定 python脚本 初始化一个Corpus"""
        dm = DatasetManager.from_config(path=self.ner_local_python_script)
        self._select_and_test(dm)

    def test_load_local_named_entity_recognition_dataset_dir(self):
        """可以指定 目录（含有同名脚本的） 初始化一个Corpus"""
        dm = DatasetManager.from_config(path=self.ner_local_dir)
        self._select_and_test(dm)

    def test_load_remote_named_entity_recognition_dataset_zip(self):
        """可以指定 dataset文件 从远端拉取数据并初始化一个Corpus"""
        dm = DatasetManager.from_config(
            task=self.ner_task,
            data_file=self.ner_remote_file,
            data_type='conll',
        )
        self._select_and_test(dm, True)

    def test_load_modelscope_named_entity_recognition_dataset(self):
        """可以指定 dataset 名称从 modelscope 拉取数据并初始化一个Corpus"""
        dm = DatasetManager.from_config(
            name=self.ner_remote_corpus_id,
            task=self.ner_task,
            namespace='izhx404',
            subset_name='adaseq',
        )
        self._select_and_test(dm)

    def test_load_huggingface_named_entity_recognition_dataset(self):
        """可以拉取 huggingface style 数据并初始化一个Corpus"""
        dm = DatasetManager.from_config(
            path=os.path.join(self.ner_local_dir, 'toy_msra_hf.py'),
            task=self.ner_task,
            transform=dict(name='hf_ner_to_adaseq'),
        )
        self._select_and_test(dm, field_strict=False)


if __name__ == '__main__':
    unittest.main()
