from typing import Dict, List, Union

from torch.utils.data import Dataset


def get_labels(dataset: Union[Dataset, Dict[str, Dataset]]) -> List[str]:
    labels = []
    if isinstance(dataset, dict):
        for _dataset in dataset.values():
            labels.extend(get_labels(_dataset))
    else:
        for data in dataset:
            labels.extend(data['labels'])
    labels = sorted(set(labels))
    if 'O' in labels:
        labels.remove('O')
        labels = ['O'] + labels
    return labels
