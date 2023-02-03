# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os.path as osp
from functools import partial
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DownloadManager
from datasets import load_dataset as hf_load_dataset
from datasets.utils.file_utils import is_remote_url
from modelscope.msdatasets import MsDataset

from adaseq.metainfo import Tasks, get_member_set

from .utils import COUNT_LABEL_FUNCTIONS, DATASET_TRANSFORMS

BUILTIN_TASKS = get_member_set(Tasks)
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    An `DatasetManager` is used by trainers to load a dataset from `modelscope`,
    local files by built-in `DatasetBuilder`s, or custom huggingface-style
    python loading scripts. It finally store a dict object with
    `train`, `valid`, `test` splits of `datasets.Dataset`.

    Args:

    datasets: `Dict[str, Dataset]`, required
        A dict that values are `datasets.Dataset` objects, keys are in `train`,
        `valid`, `test`.
    labels: `Union[str, List[str], Dict[str, Any]]`
        It could be a list of labels, e.g., `['O', 'B-LOC', 'I-LOC', ...]` .
        It could be a path or url to load the label list.
        It could be a kwargs dict to count labels from `train` and `valid` sets,
        e.g., `{'type': 'count_labels', 'key': 'label'}`.
        Types are in `COUNT_LABEL_FUNCTIONS`, you can write your own functions
        and append to `COUNT_LABEL_FUNCTIONS`.
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        labels: Optional[Union[str, List[str], Dict[str, Any]]] = None,
    ) -> None:
        self.datasets = datasets

        if labels is None:
            labels = None
        elif isinstance(labels, list):
            pass
        elif isinstance(labels, str):
            if is_remote_url(labels):
                labels = DownloadManager().download(labels)
            labels = [line.strip() for line in open(labels)]  # type: ignore
        elif isinstance(labels, dict):
            labels = labels.copy()
            label_set = set()
            counter_type = labels.pop('type')
            func = COUNT_LABEL_FUNCTIONS[counter_type]
            counter = partial(func, labels=label_set, **labels)
            kwargs = dict(desc=f'Counting labels by {counter_type}', load_from_cache_file=False)
            datasets['train'].map(counter, **kwargs)
            if 'valid' in datasets:
                datasets['valid'].map(counter, **kwargs)
            labels = sorted(label_set)
        else:
            raise ValueError(f'Unsupported labels: {labels}')
        self.labels = labels

    @property
    def train(self):  # noqa: D102
        return self.datasets.get('train', None)

    @property
    def dev(self):  # noqa: D102
        return self.datasets.get('valid', None)

    @property
    def valid(self):  # noqa: D102
        return self.datasets.get('valid', None)

    @property
    def test(self):  # noqa: D102
        return self.datasets.get('test', None)

    @classmethod
    def from_config(
        cls,
        task: Optional[str] = None,
        data_file: Optional[Union[str, Dict[str, str]]] = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
        access_token: Optional[str] = None,
        labels: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        transform: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> 'DatasetManager':
        """
        load dataset and construct an `DatasetManager`.

        Args:

        task: `str`, optional (default = `None`)
            Specifies the task when loading a dataset with built-in `DatasetBuilder`s.
            Now we have `NamedEntityRecognitionDatasetBuilder` and
            `EntityTypingDatasetBuilder`.
            It could also be the absolute path of a custom python script for the
            `datasets.load_dataset`.

        data_file: `Union[str, Dict[str, str]]`, optional (default = `None`)
            Used only when loading a dataset with built-in `DatasetBuilder`s.
            It could be an url like `"https://data.deepai.org/conll2003.zip"`,
            or a local directory (absolute path) like `"/home/data/conll2003"`,
            or a local archive file absolute path like `"/home/data/conll2003.zip"`.
            Details refer to `./dataset_builders/base.py`.
            It could also be a dict that specifies file paths (absolute) for each
            dataset splits, for example:
            `{'train': '/home/data/train.txt', 'valid': '/home/data/valid.txt'}`, or
            `{'train': 'http://DOMAIN/train.txt', 'valid': 'http://DOMAIN/valid.txt'}`.

        path: `str`,  optional (default = `None`)
            It is the name of a huggingface-hosted dataset or the absolute path
            of a custom python script for the `datasets.load_dataset`.

        name: `str`,  optional (default = `None`)
            If `data_file` or `path` are given, it is the `name` argument in
            `datasets.load_dataset`.
            If not, it is used as a modelscope dataset name.

        access_token: `str`, optional (default = `None`)
            If given, use this token to login modelscope, then we can access some
            private datasets.

        transform: `Dict[str, Any]` (default = `None`)
            function kwargs for reformat datasets, see `apply_transform()`.
        """
        if data_file is not None:
            if isinstance(data_file, str):
                if not is_remote_url(data_file) and not osp.exists(data_file):
                    raise RuntimeError('`data_file` not exists: %s', data_file)
                kwargs.update(data_dir=data_file)
            elif isinstance(data_file, dict):
                for k, v in data_file.items():
                    if is_remote_url(v):
                        continue
                    if not osp.exists(v):
                        raise RuntimeError('`data_file[%s]` not exists: %s', k, v)
                    if not osp.isabs(v):
                        # since datasets
                        raise RuntimeError('`data_file[%s]` must be a absolute path: %s', k, v)

                # we rename all `dev` key to `valid`
                if 'dev' in data_file:
                    data_file['valid'] = data_file.pop('dev')
                kwargs.update(data_files=data_file)
            else:
                raise ValueError(f'Unsupported data_file: {data_file}')

            assert task is not None and task in BUILTIN_TASKS, 'Need a specific task!'
            # where we have some pre-defined dataset builders
            task_builder_name = task
            if task in {Tasks.word_segmentation, Tasks.part_of_speech}:
                task_builder_name = Tasks.named_entity_recognition

            if task == Tasks.entity_typing and 'cand' in data_file:
                task_builder_name = 'mcce-entity-typing'

            path = osp.join(
                osp.dirname(osp.abspath(__file__)),
                'dataset_builders',
                task_builder_name.replace('-', '_') + '_dataset_builder.py',
            )

        if isinstance(path, str):
            if path.endswith('.py') or osp.isdir(path):
                logger.info('Will use a custom loading script: %s', path)

            if name is not None:
                logger.info("Passing `name='%s'` to `datasets.load_dataset`", name)

            hfdataset = hf_load_dataset(path, name=name, **kwargs)
            datasets = {k: v for k, v in hfdataset.items()}

        elif isinstance(name, str):
            # to access private datasets from modelscope
            if access_token is not None:
                from modelscope.hub.api import HubApi

                HubApi().login(access_token)

            msdataset = MsDataset.load(name, **kwargs)
            datasets = {k: v._hf_ds for k, v in msdataset.items()}

        else:
            raise RuntimeError('Unsupported dataset!')

        if 'dev' in datasets and 'valid' not in datasets:
            datasets['valid'] = datasets.pop('dev')

        if 'test' in datasets and 'valid' not in datasets:
            datasets['valid'] = datasets['test']
            logger.warning('Validation set not found. Reuse test set for validation!')

        if 'valid' in datasets and 'test' not in datasets:
            datasets['test'] = datasets['valid']
            logger.warning('Test set not found. Reuse validation set for testing!')

        if 'train' not in datasets:
            logger.warning('Training set not found!')

        if 'valid' not in datasets:
            logger.warning('Validation set not found!')

        if 'test' not in datasets:
            logger.warning('Test set not found!')

        # apply transform
        if transform:
            datasets = apply_transform(datasets, **transform)

        if labels is None:
            if task in {
                Tasks.named_entity_recognition,
                Tasks.word_segmentation,
                Tasks.part_of_speech,
                Tasks.entity_typing,
            }:
                labels = dict(type='count_span_labels')
            elif task == Tasks.relation_extraction:
                labels = dict(type='count_labels')

        if 'train' in datasets:
            logger.info('First sample in train set: ' + str(datasets['train'][0]))

        return cls(datasets, labels)  # type: ignore


def apply_transform(
    datasets: Dict[str, Dataset],
    name: str,
    **kwargs,
) -> Dict[str, Dataset]:
    """
    Apply a given function to reformat a `Dataset`.
    """
    if name not in DATASET_TRANSFORMS:
        raise RuntimeError(f'{name} not in {DATASET_TRANSFORMS.keys()}')
    kwargs = kwargs or dict()

    for k in datasets.keys():
        datasets[k] = DATASET_TRANSFORMS[name](datasets[k], **kwargs)

    return datasets
