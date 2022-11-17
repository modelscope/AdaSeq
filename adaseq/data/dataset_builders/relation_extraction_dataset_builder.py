# Copyright (c) Alibaba, Inc. and its affiliates.

import datasets
from datasets import Features, Value

from adaseq.data.dataset_builders.dataset_reader import RelationExtractionDatasetReader  # yapf: disable
from .base import CustomDatasetBuilder


class RelationExtractionDatasetBuilderConfig(datasets.BuilderConfig):
    """ Builder Config for Relation Extraction """

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super().__init__(data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class RelationExtractionDatasetBuilder(CustomDatasetBuilder):
    """ Dataset Builder for Relation Extraction """

    BUILDER_CONFIG_CLASS = RelationExtractionDatasetBuilderConfig

    def stub():  # noqa: D102
        pass

    @classmethod
    def parse_label(cls, data):  # noqa: D102
        return [data['label']]  # TODO fix

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
            return RelationExtractionDatasetReader.load_data_file(filepath, self.config.corpus_config)
