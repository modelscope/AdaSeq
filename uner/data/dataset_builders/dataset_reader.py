from abc import ABC, abstractmethod
from typing import Any, Dict

import json

from uner.data.constant import (
    NONE_REL_LABEL,
    OBJECT_START_TOKEN,
    PAD_LABEL,
    SUBJECT_START_TOKEN,
)


class DatasetReader(ABC):

    @classmethod
    @abstractmethod
    def load_data_file(cls, file_path, corpus_config):
        raise NotImplementedError


class NamedEntityRecognitionDatasetReader(DatasetReader):

    @classmethod
    def load_data_file(cls, file_path, corpus_config):
        if corpus_config['data_type'] == 'sequence_labeling':
            if corpus_config['data_format'] == 'column':
                return cls._load_column_data_file(
                    file_path, delimiter=corpus_config.get('delimiter', None))
            elif corpus_config['data_format'] == 'json':
                return cls._load_sequence_labeling_json_data_file(
                    file_path, corpus_config)
        elif corpus_config['data_type'] == 'span_based':
            return cls._load_span_based_json_data_file(
                file_path, corpus_config.get('is_end_included', False))
        else:
            raise ValueError('Unknown corpus format type [%s]'
                             % corpus_config['data_type'])

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
                            'mask': mask
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
                yield guid, {
                    'id': str(guid),
                    'tokens': tokens,
                    'spans': spans,
                    'mask': mask
                }

    @classmethod
    def _load_sequence_labeling_json_data_file(cls, filepath, corpus_config):
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
                labels = example['labels']
                assert len(tokens) == len(labels)
                spans = cls._labels_to_spans(labels)
                mask = cls._labels_to_mask(labels)
                yield guid, {
                    'id': str(guid),
                    'tokens': tokens,
                    'spans': spans,
                    'mask': mask
                }
                guid += 1

    @classmethod
    def _load_span_based_json_data_file(cls, filepath, corpus_config):
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
                entity_list = []
                entities = example['label']
                for entity_type, span_list in entities.items():
                    for name, span in span_list.items():
                        end_offset = 0
                        if corpus_config['is_end_included'] is True:
                            end_offset = 1
                        entity_list.append({
                            'start': span[0][0],
                            'end': span[0][1] + end_offset,
                            'type': entity_type
                        })
                mask = [True] * len(tokens)
                yield guid, {
                    'id': str(guid),
                    'tokens': tokens,
                    'spans': entity_list,
                    'mask': mask
                }
                guid += 1

    @classmethod
    def _labels_to_spans(cls, labels):
        spans = []
        in_entity = False
        start = -1
        for i in range(len(labels)):
            # fix label error
            if labels[i][0] in 'IE' and not in_entity:
                labels[i] = 'B' + labels[i][1:]
            if labels[i][0] in 'BS':
                if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                    start = i
                else:
                    spans.append({
                        'start': i,
                        'end': i + 1,
                        'type': labels[i][2:]
                    })
            elif labels[i][0] in 'IE':
                if i + 1 >= len(labels) or labels[i + 1][0] not in 'IE':
                    assert start >= 0, \
                        'Invalid label sequence found: {}'.format(labels)
                    spans.append({
                        'start': start,
                        'end': i + 1,
                        'type': labels[i][2:]
                    })
                    start = -1
            if labels[i][0] in 'B':
                in_entity = True
            elif labels[i][0] in 'OES':
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
                                         e.g., text = 'This is an example.',
                                               mention = 'example',
                                               end = 3 if is_end_included is True.

        """

        with open(file_path, encoding='utf-8') as f:
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
                entity_list = []
                entities = example['label']
                for entity in entities:
                    end_offset = 0
                    if corpus_config.get('is_end_included', False) is True:
                        end_offset = 1
                    entity_list.append({
                        'start': entity['start'],
                        'end': entity['end'] + end_offset,
                        'type': entity['type']
                    })
                mask = [True] * len(tokens)
                yield guid, {
                    'id': str(guid),
                    'tokens': tokens,
                    'spans': entity_list,
                    'mask': mask
                }
                guid += 1


class RelationExtractionDatasetReader(DatasetReader):

    @classmethod
    def load_data_file(cls, file_path, corpus_config):
        if corpus_config['data_type'] == 'sequence_labeling':
            if corpus_config['data_format'] == 'column':
                return cls._load_column_data_file(
                    file_path, delimiter=corpus_config.get('delimiter', None))
        else:
            raise ValueError('Unknown corpus format type [%s]'
                             % corpus_config['data_type'])

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
