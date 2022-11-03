import os

import datasets
from datasets import Features, Value

from uner.data.dataset_builders.dataset_reader import \
    EntityTypingDatasetReader  # yapf: disable
from .base import CustomDatasetBuilder


class EntityTypingDatasetBuilderConfig(datasets.BuilderConfig):

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super(EntityTypingDatasetBuilderConfig, self).__init__(
            data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class EntityTypingDatasetBuilder(CustomDatasetBuilder):

    BUILDER_CONFIG_CLASS = EntityTypingDatasetBuilderConfig

    def stub():
        pass

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features({
                'id':
                Value('string'),
                'tokens': [Value('string')],
                'spans': [{
                    'start': Value('int32'),  # close
                    'end': Value('int32'),  # open
                    'type': [Value('string')]
                }],
                'mask': [Value('bool')]
            }))
        return info

    def _generate_examples(self, filepath):
        if 'corpus_reader' in self.config.corpus_config:
            # TODO: get the reder via reflection
            raise NotImplementedError
        else:
            return EntityTypingDatasetReader.load_data_file(
                filepath, self.config.corpus_config)

    @classmethod
    def parse_label(cls, data):
        labels = []
        for entity in data['spans']:
            labels.extend(entity['type'])
        return set(labels)
