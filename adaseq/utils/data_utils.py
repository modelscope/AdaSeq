# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Union

from torch.utils.data import Dataset

from .common_utils import import_dataset_builder_class


def get_labels(dataset: Union[Dataset, Dict[str, Dataset]]) -> List[str]:
    """ [deprecated] Collect label set from datasets

    Args:
        dataset (Union[Dataset, Dict[str, Dataset]): dataset or a dict of datasets

    Returns:
        labels (List[str]): list of all unique labels in alphabetical order
    """
    labels = []
    if isinstance(dataset, dict):
        for _dataset in dataset.values():
            labels.extend(get_labels(_dataset))
    else:
        for data in dataset:
            builder_name = dataset.info.builder_name
            builder_cls = import_dataset_builder_class(builder_name)
            labels.extend(builder_cls.parse_label(data))
    labels = sorted(set(labels))
    return labels


def gen_label2id(labels: List[str]) -> Dict[str, int]:
    """ Generate label2id from labels

    Args:
        labels (List[str]): list of all labels

    Returns:
        label2id (Dict[str, int]): a dict mapping label string to id
    """
    label2id = {}
    for label in labels:
        label2id[label] = len(label2id)
    return label2id
