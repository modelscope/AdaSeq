import os
from typing import Optional
import yaml

from datasets import load_dataset


class Corpus:
    def __init__(self,
                 task: str,
                 corpus: Optional[str] = None,
                 train: Optional[str] = None,
                 dev: Optional[str] = None,
                 valid: Optional[str] = None,
                 test: Optional[str] = None,
                 **kwargs):
        self._init_predefined_corpus_config()

        task = task.replace('-', '_')
        data_dir = None
        data_files = None

        if corpus is not None:
            data_dir = self._get_predefined_corpus_url_by_name(task, corpus)
        else:
            assert train is not None \
                or dev is not None \
                or valid is not None \
                or test is not None
            data_files = {}
            if train is not None:
                data_files['train'] = train
            if dev is not None and valid is None:
                valid = dev
            if valid is not None:
                data_files['valid'] = valid
            if test is not None:
                data_files['test'] = test

        self.datasets = load_dataset(
            f'uner/datasets/dataset_builders/{task}_dataset_builder.py',
            data_dir=data_dir,
            data_files=data_files)

    @property
    def train(self):
        return self.datasets.get('train', None)

    @property
    def dev(self):
        return self.datasets.get('valid', None)

    @property
    def valid(self):
        return self.datasets.get('valid', None)

    @property
    def test(self):
        return self.datasets.get('test', None)

    def _init_predefined_corpus_config(self):
        config_file = os.path.join('uner', 'datasets', 'corpus.yaml')
        with open(config_file, 'r') as f:
            self._predefined_corpus_config = yaml.safe_load(f)

    def _get_predefined_corpus_url_by_name(self, task, name):
        return self._predefined_corpus_config[task][name]['url']

