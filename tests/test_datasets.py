import os
import unittest

from adaseq.data.dataset_manager import DatasetManager


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.setUp_ner_dataset()
        self.setUp_typing_dataset()

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

    def _select_and_test(self, dm: DatasetManager, mask: bool = False):
        if mask:
            ner_expected_data = self.ner_expected_data
        else:
            ner_expected_data = self.ner_expected_data.copy()
            ner_expected_data.pop('mask')

        selected_data = dm.train[int(ner_expected_data['id'])]
        self.assertEqual(len(dm.train), self.ner_train_data_size)
        self.assertEqual(len(dm.test), self.ner_test_data_size)
        self.assertEqual(selected_data, ner_expected_data)

    def test_load_local_named_entity_recognition_dataset_files(self):
        """可以指定train文件初始化一个Corpus"""
        dataset = DatasetManager(
            task=self.ner_task,
            data_files=dict(train=self.ner_local_train_file, test=self.ner_local_test_file),
            data_type='sequence_labeling',
            data_format='column',
        )
        self._select_and_test(dataset, True)

    def test_load_remote_named_entity_recognition_dataset_files(self):
        """可以指定远程 train 文件初始化一个Corpus"""
        dataset = DatasetManager(
            task=self.ner_task,
            data_files=self.ner_remote_splits,
            data_type='sequence_labeling',
            data_format='column',
        )
        self._select_and_test(dataset, True)

    def test_load_local_named_entity_recognition_dataset_zip(self):
        """可以指定 dataset文件 从本地初始化一个Corpus"""
        dataset = DatasetManager(
            task=self.ner_task,
            data_dir=self.ner_local_zip_file,
            data_type='sequence_labeling',
            data_format='column',
        )
        self._select_and_test(dataset, True)

    def test_load_local_named_entity_recognition_dataset_script(self):
        """可以指定 python脚本 初始化一个Corpus"""
        dataset = DatasetManager(name_or_path=self.ner_local_python_script)
        self._select_and_test(dataset)

    def test_load_local_named_entity_recognition_dataset_dir(self):
        """可以指定 目录（含有同名脚本的） 初始化一个Corpus"""
        dataset = DatasetManager(name_or_path=self.ner_local_dir)
        self._select_and_test(dataset)

    def test_load_remote_named_entity_recognition_dataset_zip(self):
        """可以指定 dataset文件 从远端拉取数据并初始化一个Corpus"""
        dataset = DatasetManager(
            task=self.ner_task,
            data_dir=self.ner_remote_file,
            data_type='sequence_labeling',
            data_format='column',
        )
        self._select_and_test(dataset, True)

    def test_load_modelscope_named_entity_recognition_dataset(self):
        """可以指定 dataset 名称从 modelscope 拉取数据并初始化一个Corpus"""
        dataset = DatasetManager(
            name_or_path=self.ner_remote_corpus_id, task=self.ner_task, namespace='izhx404'
        )
        self._select_and_test(dataset)

    def setUp_typing_dataset(self):
        self.typing_task = 'entity-typing'
        self.typing_remote_corpus_id = 'toy_wiki'
        self.typing_train_data_size = 1000
        self.typing_test_data_size = 200
        self.typing_expected_data = {
            'id': '4',
            'tokens': [
                '并',
                '请',
                '莫',
                '文',
                '祥',
                '同',
                '志',
                '向',
                '其',
                '他',
                '中',
                '央',
                '领',
                '导',
                '同',
                '志',
                '转',
                '达',
                '这',
                '个',
                '意',
                '见',
                '。',
            ],
            'spans': [{'start': 2, 'end': 5, 'type': ['人', '领导', '人', '男性', '男人', '同事']}],
            'mask': [True] * 23,
        }


if __name__ == '__main__':
    unittest.main()
