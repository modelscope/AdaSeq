# Copyright (c) Alibaba, Inc. and its affiliates.

import datasets
from datasets import Features, Value

from adaseq.data.dataset_builders.dataset_reader import (
    NamedEntityRecognitionDatasetReader,
)

from .base import CustomDatasetBuilder


class PartOfSpeechDatasetBuilderConfig(datasets.BuilderConfig):
    """Builder Config for the Part of Speech task"""

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super().__init__(data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class PartOfSpeechDatasetBuilder(CustomDatasetBuilder):
    """Dataset Builder for the Part of Speech task"""

    BUILDER_CONFIG_CLASS = PartOfSpeechDatasetBuilderConfig

    def stub():  # noqa: D102
        pass

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features(
                {
                    'id': Value('string'),
                    'tokens': [Value('string')],
                    'spans': [
                        {
                            'start': Value('int32'),  # close
                            'end': Value('int32'),  # open
                            'type': Value('string'),
                        }
                    ],
                    'mask': [Value('bool')],
                }
            )
        )
        return info

    def _generate_examples(self, filepath):
        if 'corpus_reader' in self.config.corpus_config:
            # TODO: get the reader via reflection
            raise NotImplementedError
        else:
            return NamedEntityRecognitionDatasetReader.load_data_file(
                filepath, self.config.corpus_config
            )
