# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from adaseq.data.constant import (
    NONE_REL_LABEL,
    OBJECT_START_TOKEN,
    PAD_LABEL,
    SUBJECT_START_TOKEN,
)


class DatasetReader(ABC):
    """DatasetReader abstract class"""

    @classmethod
    @abstractmethod
    def load_data_file(cls, file_path, corpus_config):
        """Reads the instances from the given `file_path` and `corpus_config`"""
        raise NotImplementedError


class NamedEntityRecognitionDatasetReader(DatasetReader):
    """Implementation of NER reader."""

    @classmethod
    def load_data_file(cls, file_path, corpus_config):
        """
        NER reader supports:
        1. CoNLL format ('column'),
        2. json format
        ```
        {
            'text': 'duck duck duck duck',
            'labels': ['B-PER', 'O', ...]
        }
        ```
        3. json spans format
        ```
        {
            'text': 'duck duck',
            'spans': [
                {'start': 0, 'end': 1, 'type': 'PER'},
                ...
            ]
        ```
        4. cluener format
        ```
        {
            'text': 'duck duck',
            'label': {
                'LOC': [[0, 1], ...]
            }
        }
        ```.
        """
        if corpus_config['data_type'] == 'sequence_labeling':
            if corpus_config['data_format'] == 'column':
                return cls._load_column_data_file(
                    file_path, delimiter=corpus_config.get('delimiter', None)
                )
            elif corpus_config['data_format'] == 'json':
                return cls._load_sequence_labeling_json_data_file(file_path, corpus_config)
        elif corpus_config['data_type'] == 'json_spans':
            return cls._load_json_spans_data_file(file_path, corpus_config)
        elif corpus_config['data_type'] == 'cluener':
            return cls._load_cluener_json_data_file(
                file_path, corpus_config.get('is_end_included', False)
            )
        else:
            raise ValueError('Unknown corpus format type [%s]' % corpus_config['data_type'])

    @classmethod
    def _load_column_data_file(cls, file_path, delimiter):
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
    def _load_sequence_labeling_json_data_file(cls, filepath, corpus_config):
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
    def _load_json_spans_data_file(cls, filepath, corpus_config):
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
                    spans.append(span)
                mask = [True] * len(tokens)
                yield guid, {'id': str(guid), 'tokens': tokens, 'spans': spans, 'mask': mask}
                guid += 1

    @classmethod
    def _load_cluener_json_data_file(cls, filepath, corpus_config):
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


class EntityTypingDatasetReader(DatasetReader):
    """Entity typing dataset reader."""

    @classmethod
    def load_data_file(cls, file_path: str, corpus_config: Dict[str, Any]):
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
        spans_key = corpus_config.get('spans_key', 'label')
        text_key = corpus_config.get('text_key', 'text')
        tokenizer = corpus_config.get('tokenizer', 'char')
        is_end_included = corpus_config.get('is_end_included', False)

        with open(file_path, encoding='utf-8') as f:
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


class RelationExtractionDatasetReader(DatasetReader):
    """Relation Extraction dataset reader"""

    @classmethod
    def load_data_file(cls, file_path, corpus_config):
        """load CoNLL format file."""
        if corpus_config['data_type'] == 'sequence_labeling':
            if corpus_config['data_format'] == 'column':
                return cls._load_column_data_file(
                    file_path, delimiter=corpus_config.get('delimiter', None)
                )
        else:
            raise ValueError('Unknown corpus format type [%s]' % corpus_config['data_type'])

    @classmethod
    def _load_column_data_file(cls, file_path, delimiter):
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
