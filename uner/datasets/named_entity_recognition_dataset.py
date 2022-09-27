import os
from typing import Optional

import yaml
from datasets import load_dataset


class NamedEntityRecognitionDataset:
   
    TARGET_TASK_NAME = 'name_entity_recognition'

    def __init__(self,
                 **corpus_config):
        self._init_predefined_corpus_config()

        data_dir = None
        data_files = None

        if 'name' in corpus_config:
            data_dir = self._get_predefined_corpus_url_by_name(TARGET_TASK_NAME, corpus_config['name'])
        else:
            data_files = {}
            for key in ['train', 'dev', 'valid', 'test']:
                if key in corpus_config:
                    data_files[key] = corpus_config[key.replace('dev', 'valid')]
            assert len(data_files) > 0 

        self.datasets = load_dataset(
            f'uner/datasets/dataset_builders/name_entity_recognition_dataset_builder.py',
            data_dir=data_dir,
            data_files=data_files,
            corpus_config=corpus_config)  # tell the builder how to reader corpus files. if someone wants to use a custumized reader other than column based or json based, he can pass a reader function. 

        if self.valid is None and self.test is not None:
            self.datasets['valid'] = self.datasets['test']

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
