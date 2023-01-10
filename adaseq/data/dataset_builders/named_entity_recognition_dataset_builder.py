# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from typing import Dict

import datasets
from datasets import Features, Value

from adaseq.data.constant import PAD_LABEL

from .base import CustomDatasetBuilder


class NamedEntityRecognitionDatasetBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for Named Entity Recognition datasets"""

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super().__init__(data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class NamedEntityRecognitionDatasetBuilder(CustomDatasetBuilder):
    """Builder for entity typing datasets.

    features:
        id: string, data record id.
        tokens: list[str] input tokens.
        spans: List[Dict],  mentions like: [{'start': 0, 'end': 2, 'type': 'PER'}]
        mask: bool, mention mask.
    """

    BUILDER_CONFIG_CLASS = NamedEntityRecognitionDatasetBuilderConfig

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
        """
        NER reader supports:
        1. data_type: conll
            ```
            duck B-PER
            duck I-PER
            duck O
            ```

        2. data_type: json_tags
        ```
        {
            'text': 'duck duck duck duck',
            'labels': ['B-PER', 'O', ...]
        }
        ```

        3. data_type: json_spans
        ```
        {
            'text': 'duck duck',
            'spans': [
                {'start': 0, 'end': 1, 'type': 'PER'},
                ...
            ]
        ```

        4. data_type: cluener
        ```
        {
            'text': 'duck duck',
            'label': {
                'LOC': [[0, 1], ...]
            }
        }
        ```
        """
        corpus_config = self.config.corpus_config
        if corpus_config['data_type'] == 'conll':
            return self._load_conll_file(filepath, corpus_config)
        elif corpus_config['data_type'] == 'json_tags':
            return self._load_json_tags_file(filepath, corpus_config)
        elif corpus_config['data_type'] == 'json_spans':
            return self._load_json_spans_file(filepath, corpus_config)
        elif corpus_config['data_type'] == 'cluener':
            return self._load_cluener_file(filepath, corpus_config)
        else:
            raise ValueError('Unknown corpus format type [%s]' % corpus_config['data_type'])

    @classmethod
    def _load_conll_file(cls, file_path, corpus_config: Dict):
        delimiter = corpus_config.get('delimiter', None)
        with open(file_path, encoding='utf-8') as f:
            guid = 0
            tokens = []
            labels = []
            for line in f:
                if line.startswith('-DOCSTART-') or line == '' or line == '\n':
                    if tokens:
                        spans = cls._labels_to_spans(labels)
                        mask = cls._labels_to_mask(labels)
                        yield guid, {
                            'id': str(guid),
                            'tokens': tokens,
                            'spans': spans,
                            'mask': mask,
                        }
                        guid += 1
                        tokens = []
                        labels = []
                else:
                    splits = line.split(delimiter)
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())
            if tokens:
                spans = cls._labels_to_spans(labels)
                mask = cls._labels_to_mask(labels)
                yield guid, {'id': str(guid), 'tokens': tokens, 'spans': spans, 'mask': mask}

    @classmethod
    def _load_json_tags_file(cls, filepath, corpus_config):
        tags_key = corpus_config.get('tags_key', 'labels')
        text_key = corpus_config.get('text_key', 'text')
        tokenizer = corpus_config.get('tokenizer', 'char')
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
                        tokens = text.split(' ')
                    else:
                        raise NotImplementedError
                else:
                    raise ValueError('Unsupported text input.')
                labels = example[tags_key]
                assert len(tokens) == len(labels)
                spans = cls._labels_to_spans(labels)
                mask = cls._labels_to_mask(labels)
                yield guid, {'id': str(guid), 'tokens': tokens, 'spans': spans, 'mask': mask}
                guid += 1

    @classmethod
    def _load_json_spans_file(cls, filepath, corpus_config):
        # {'text': 'aaa', 'labels': [{'start': 0, 'end': 1, type: 'LOC'}, ...]}
        # {'tokens': ['a', 'aa', ...], 'spans': [{'start': 0, 'end': 1, type: 'LOC'}, ...]}
        spans_key = corpus_config.get('spans_key', 'spans')
        text_key = corpus_config.get('text_key', 'text')
        tokenizer = corpus_config.get('tokenizer', 'char')
        is_end_included = corpus_config.get('is_end_included', False)

        with open(filepath, encoding='utf-8') as f:
            guid = 0
            for line in f:
                if line.strip() == '':
                    continue
                example = json.loads(line)
                text = example[text_key]
                if isinstance(text, list):
                    tokens = text
                elif isinstance(text, str):
                    if tokenizer == 'char':
                        tokens = list(text)
                    elif tokenizer == 'blank':
                        tokens = text.split(' ')
                    else:
                        raise RuntimeError
                else:
                    raise ValueError('Unsupported text input.')
                spans = []
                for span in example[spans_key]:
                    if is_end_included:
                        span['end'] += 1
                    if 'word' in span:
                        del span['word']
                    spans.append(span)
                mask = [True] * len(tokens)
                yield guid, {'id': str(guid), 'tokens': tokens, 'spans': spans, 'mask': mask}
                guid += 1

    @classmethod
    def _load_cluener_file(cls, filepath, corpus_config):
        is_end_included = corpus_config.get('is_end_included', False)

        with open(filepath, encoding='utf-8') as f:
            guid = 0
            for line in f:
                example = json.loads(line)
                text = example['text']
                if isinstance(text, list):
                    tokens = text
                elif isinstance(text, str):
                    if corpus_config['tokenizer'] == 'char':
                        tokens = list(text)
                    elif corpus_config['tokenizer'] == 'blank':
                        tokens = text.split(' ')
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                entities = list()
                for entity_type, span_list in example['label'].items():
                    for name, span in span_list.items():
                        end_offset = int(is_end_included)
                        span = dict(start=span[0][0], end=span[0][1] + end_offset, type=entity_type)
                        entities.append(span)
                mask = [True] * len(tokens)
                yield guid, {'id': str(guid), 'tokens': tokens, 'spans': entities, 'mask': mask}
                guid += 1

    @classmethod
    def _labels_to_spans(cls, labels):
        spans = []
        in_entity = False
        start = -1
        for i, tag in enumerate(labels):
            # fix label error
            if tag[0] in 'IE' and not in_entity:
                tag = 'B' + tag[1:]
            if tag[0] in 'BS':
                if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                    start = i
                else:
                    spans.append(dict(start=i, end=i + 1, type=tag[2:]))
            elif tag[0] in 'IE':
                if i + 1 >= len(labels) or labels[i + 1][0] not in 'IE':
                    assert start >= 0, 'Invalid label sequence found: {}'.format(labels)
                    spans.append(dict(start=start, end=i + 1, type=tag[2:]))
                    start = -1
            if tag[0] in 'B':
                in_entity = True
            elif tag[0] in 'OES':
                in_entity = False
        return spans

    @classmethod
    def _labels_to_mask(cls, labels):
        mask = []
        for label in labels:
            mask.append(label != PAD_LABEL)
        return mask
