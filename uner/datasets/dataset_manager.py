import os
from typing import Optional

import yaml
from datasets import load_dataset


class DatasetManager:

    def __init__(self,
                 task: str,
                 corpus: Optional[str] = None,
                 train: Optional[str] = None,
                 dev: Optional[str] = None,
                 valid: Optional[str] = None,
                 test: Optional[str] = None,
                 **corpus_config):
        self._init_predefined_corpus_config()
        data_url = None
        data_files = None

        if corpus is not None:
            pre_defined_corpus_config = self._get_predefined_corpus_config_by_name(
                task, corpus)
            if pre_defined_corpus_config is not None:
                for k, v in pre_defined_corpus_config.items():
                    if k not in corpus_config:
                        corpus_config[k] = v

        if 'url' in corpus_config:
            data_url = corpus_config.pop('url')
        else:
            data_files = {}
            for split in ['train', 'dev', 'valid', 'test']:
                data_file = locals()[split]
                if data_file is not None:
                    data_files[split.replace('dev', 'valid')] = data_file
            assert len(data_files) > 0

        _task = task.replace('-', '_')
        self.datasets = load_dataset(
            f'uner/datasets/dataset_builders/{_task}_dataset_builder.py',
            data_dir=data_url,
            data_files=data_files,
            **corpus_config
        )  # tell the builder how to reader corpus files. if someone wants to use a custumized reader other than column based or json based, he can pass a reader function.  # noqa

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

    def _get_predefined_corpus_config_by_name(self, task, name):
        return self._predefined_corpus_config[task][name]
