# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import datasets
from datasets import Features, Value

from uner.data.dataset_builders.dataset_reader import \
    RelationExtractionDatasetReader  # yapf: disable
from .base import CustomDatasetBuilder


class RelationExtractionDatasetBuilderConfig(datasets.BuilderConfig):

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super(RelationExtractionDatasetBuilderConfig, self).__init__(
            data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class RelationExtractionDatasetBuilder(CustomDatasetBuilder):
    BUILDER_CONFIG_CLASS = RelationExtractionDatasetBuilderConfig

    def stub():
        pass

    @classmethod
    def parse_label(cls, data):
        return [data['label']]

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features({
                'id': Value('string'),
                'tokens': [Value('string')],
                'mask': [Value('bool')],
                'so_head_mask': [Value('bool')],
                'label': Value('string'),
            }))
        return info

    def _generate_examples(self, filepath):
        if 'corpus_reader' in self.config.corpus_config:
            # TODO: get the reder via reflection
            raise NotImplementedError
        else:
            return RelationExtractionDatasetReader.load_data_file(
                filepath, self.config.corpus_config)
