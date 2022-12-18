# Copyright (c) Alibaba, Inc. and its affiliates.

import datasets
from datasets import Features, Value

from adaseq.data.constant import (
    NONE_REL_LABEL,
    OBJECT_START_TOKEN,
    PAD_LABEL,
    SUBJECT_START_TOKEN,
)

from .base import CustomDatasetBuilder


class RelationExtractionDatasetBuilderConfig(datasets.BuilderConfig):
    """Builder Config for Relation Extraction"""

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super().__init__(data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class RelationExtractionDatasetBuilder(CustomDatasetBuilder):
    """Dataset Builder for Relation Extraction"""

    BUILDER_CONFIG_CLASS = RelationExtractionDatasetBuilderConfig

    def stub():  # noqa: D102
        pass

    @classmethod
    def parse_label(cls, data):  # noqa: D102
        return [data['label']]  # TODO fix

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features(
                {
                    'id': Value('string'),
                    'tokens': [Value('string')],
                    'mask': [Value('bool')],
                    'so_head_mask': [Value('bool')],
                    'label': Value('string'),
                }
            )
        )
        return info

    def _generate_examples(self, filepath):
        corpus_config = self.config.corpus_config
        if corpus_config['data_type'] == 'conll':
            return self._load_conll_file(filepath, corpus_config)
        else:
            raise ValueError('Unknown corpus format type [%s]' % corpus_config['data_type'])

    @classmethod
    def load_data_file(cls, file_path, corpus_config):
        """load CoNLL format file."""
        if corpus_config['data_type'] == 'conll':
            return cls._load_conll_file(file_path, corpus_config)
        else:
            raise ValueError('Unknown corpus format type [%s]' % corpus_config['data_type'])

    @classmethod
    def _load_conll_file(cls, file_path, corpus_config):
        delimiter = corpus_config.get('delimiter', None)

        with open(file_path, encoding='utf-8') as f:
            guid = 0
            tokens = []
            labels = []
            for line in f:
                if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                    if tokens:
                        mask = cls._labels_to_mask(labels)
                        so_head_mask = cls._create_so_head_mask(tokens)
                        label = cls._extract_rel_label(tokens, labels)
                        yield guid, {
                            'id': str(guid),
                            'tokens': tokens,
                            'label': label,
                            'mask': mask,
                            'so_head_mask': so_head_mask,
                        }
                        guid += 1
                        tokens = []
                        labels = []
                else:
                    splits = line.split(delimiter)
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())

            if tokens:
                mask = cls._labels_to_mask(labels)
                so_head_mask = cls._create_so_head_mask(tokens)
                label = cls._extract_rel_label(tokens, labels)
                yield guid, {
                    'id': str(guid),
                    'tokens': tokens,
                    'label': label,
                    'mask': mask,
                    'so_head_mask': so_head_mask,
                }

    @classmethod
    def _extract_rel_label(cls, tokens, labels):
        """
        example:
        token   label
        the     O
        <E>     /misc/misc/part_of
        Circle  B-A
        Undone  I-A
        </E>    O
        has     O
        been    O
        released        O
        """
        rel_label = NONE_REL_LABEL
        for token, label in zip(tokens, labels):
            if token in (SUBJECT_START_TOKEN, OBJECT_START_TOKEN):
                rel_label = label
                break
        return rel_label

    @classmethod
    def _create_so_head_mask(cls, tokens):
        mask = []
        for token in tokens:
            mask.append(token in (SUBJECT_START_TOKEN, OBJECT_START_TOKEN))
        return mask

    @classmethod
    def _labels_to_mask(cls, labels):
        mask = []
        for label in labels:
            mask.append(label != PAD_LABEL)
        return mask
