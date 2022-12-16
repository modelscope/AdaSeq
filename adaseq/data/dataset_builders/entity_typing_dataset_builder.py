# Copyright (c) Alibaba, Inc. and its affiliates.

import json

import datasets
from datasets import Features, Value

from .base import CustomDatasetBuilder


class EntityTypingDatasetBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for entity typing datasets"""

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super().__init__(data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class EntityTypingDatasetBuilder(CustomDatasetBuilder):
    """Builder for entity typing datasets.

    features:
        id: string, data record id.
        tokens: list[str] input tokens.
        spans: List[Dict],  mentions like: [{'start': 0, 'end': 2, 'type': ['PER', 'MAN']}]
        mask: bool, mention mask.
    """

    BUILDER_CONFIG_CLASS = EntityTypingDatasetBuilderConfig

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
                            'type': [Value('string')],
                        }
                    ],
                    'mask': [Value('bool')],
                }
            )
        )
        return info

    def _generate_examples(self, filepath):
        """Load data file.
        Args:
        file_path: string, data file path.
        corpus_config: dictionary, required keys:
        tokenizer: string, specify the tokenization method,
            'char' list(text) and 'blank' for text.split(' ')
        is_end_included: bool, wheather the end index pointer to the
            last token or the token next to the last token.
            e.g., text = 'This is an example.', mention = 'example',
            end = 3 if is_end_included is True.

        """
        corpus_config = self.config.corpus_config
        spans_key = corpus_config.get('spans_key', 'label')
        text_key = corpus_config.get('text_key', 'text')
        tokenizer = corpus_config.get('tokenizer', 'char')
        is_end_included = corpus_config.get('is_end_included', False)

        with open(filepath, encoding='utf-8') as f:
            guid = 0
            for line in f:
                example = json.loads(line)
                text = example[text_key]
                if isinstance(text, list):
                    tokens = text
                elif isinstance(text, str):
                    if tokenizer == 'char':
                        tokens = list(text)
                    elif tokenizer == 'blank':
                        tokens = text.split()
                    else:
                        raise NotImplementedError
                else:
                    raise ValueError('Unsupported text input.')

                entities = list()
                for span in example[spans_key]:
                    if is_end_included:
                        span['end'] += 1
                    entities.append(span)

                mask = [True] * len(tokens)

                reading_format = corpus_config.get('encoding_format', 'span')
                if reading_format == 'span':
                    yield guid, {
                        'id': str(guid),
                        'tokens': tokens,
                        'spans': entities,
                        'mask': mask,
                    }
                    guid += 1
                elif reading_format == 'concat':
                    for entity in entities:
                        yield guid, {
                            'id': str(guid),
                            'tokens': tokens,
                            'spans': [entity],
                            'mask': mask,
                        }
                        guid += 1
                else:
                    raise NotImplementedError('unimplemented reading_format')
