# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, List, Optional, Set

from datasets.arrow_dataset import Dataset

from .span_utils import bio_tags_to_spans


def count_labels(example: Dict, labels: Set, key: str = 'label', **kwargs):
    """
    Count labels from dataset.
    """
    if isinstance(example[key], str):
        labels.add(example[key])
    elif isinstance(example[key], list):
        labels.update(example[key])
    else:
        raise RuntimeError
    return example


def count_span_labels(example: Dict, labels: Set, key: str = 'spans', **kwargs):
    """
    Count labels from dataset.
    """
    for span in example[key]:
        label = span['type']
        if isinstance(label, str):
            labels.add(label)
        elif isinstance(label, list):
            labels.update(label)
        else:
            raise RuntimeError
    return example


def hf_ner_to_adaseq(
    dataset: Dataset,
    key: str = 'ner_tags',
    scheme: str = 'bio',
    classes_to_ignore: Optional[List[str]] = None,
) -> Dataset:
    """
    Map the feature with name `key` from index to label.
    """
    if scheme.lower() == 'bio':
        to_spans = bio_tags_to_spans
    else:
        raise NotImplementedError

    to_str = dataset.features[key].feature.int2str

    def to_adaseq(example):
        tags = [to_str(i) for i in example[key]]
        example['spans'] = to_spans(tags, classes_to_ignore)
        return example

    dataset = dataset.map(to_adaseq)
    return dataset


COUNT_LABEL_FUNCTIONS = {'count_labels': count_labels, 'count_span_labels': count_span_labels}
DATASET_TRANSFORMS = {'hf_ner_to_adaseq': hf_ner_to_adaseq}
