import os.path as osp
from typing import Dict, Optional

from datasets import load_dataset as hf_load_dataset
from datasets.utils.file_utils import is_remote_url
from modelscope.msdatasets import MsDataset
from modelscope.utils.logger import get_logger

SUPPORTED_LOCAL_TASK = {
    'chinese-word-segmentation', 'entity-typing', 'named-entity-recognition',
    'part-of-speech'
}
logger = get_logger(log_level='INFO')


class DatasetManager:
    """
    An `DatasetManager` is used by trainers to load a dataset from `modelscope`,
    local files by built-in `DatasetBuilder`s, or custom huggingface-style
    python loading scripts. It finally store a `datasets.Dataset` object with
    `train`, `valid`, `test` splits.

    Args:

    name_or_path: `str`,  optional (default = `None`)
        It is a modelscope dataset name, which is officially uploaded by Alibaba
        DAMO Academy with some specific pre-process operations.
        It is not used when loading local files with built-in `DatasetBuilder`s,
        now we have `NamedEntityRecognitionDatasetBuilder` and
        `EntityTypingDatasetBuilder`.
        It could also be the absolute path of a custom python script for the
        `datasets.load_dataset`.

    task: `str`, optional (default = `None`)
        Specifies the task when loading a dataset with built-in `DatasetBuilder`s.

    data_dir: `str`, optional (default = `None`)
        Used only when loading a dataset with built-in `DatasetBuilder`s.
        It could be a url like `"https://data.deepai.org/conll2003.zip"`,
        or a local directory (absolute path) like `"/home/data/conll2003"`,
        or a local archieve file absolute path like `"/home/data/conll2003.zip"`.
        Details refer to `./dataset_builders/base.py`.

    data_files: `Dict[str, str]`, optional (default = `None`)
        Used only when loading a dataset with built-in `DatasetBuilder`s.
        Specifies file paths (absolute) for each dataset splits.
        `{'train': '/home/data/train.txt', 'valid': '/home/data/valid.txt'}`, or
        `{'train': 'http://DOMAIN/train.txt', 'valid': 'http://DOMAIN/valid.txt'}`

    access_token: `str`, optional (default = `None`)
        If given, use this token to login modelscope, then we can access some
        private datasets.

    """

    def __init__(self,
                 name_or_path: Optional[str] = None,
                 task: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 data_files: Optional[Dict[str, str]] = None,
                 access_token: Optional[str] = None,
                 **kwargs):
        if data_dir or data_files:
            if isinstance(data_dir, str):
                if not is_remote_url(data_dir) and not osp.exists(data_dir):
                    raise RuntimeError('`data_dir` not exists: %s', data_dir)
            elif isinstance(data_files, dict):
                for k, v in data_files.items():
                    if not is_remote_url(data_dir) and not osp.exists(v):
                        raise RuntimeError('`data_file[%s]` not exists: %s', k,
                                           v)
                    if not osp.isabs(v):
                        # since datasets
                        raise RuntimeError(
                            '`data_file[%s]` must be a absolute path: %s', k,
                            v)

                # we rename all `dev` key to `valid`
                if 'dev' in data_files:
                    data_files['valid'] = data_files.pop('dev')

            assert task in SUPPORTED_LOCAL_TASK, 'Need a specific task!'
            # where we have some pre-defined dataset builders
            code_path = osp.join(
                osp.dirname(osp.abspath(__file__)), 'dataset_builders',
                task.replace('-', '_') + '_dataset_builder.py')
            self.datasets = hf_load_dataset(
                code_path, data_dir=data_dir, data_files=data_files, **kwargs)

        elif isinstance(name_or_path, str):
            if name_or_path.endswith('.py') or osp.isdir(name_or_path):
                # user defined
                logger.info('Will use a custom dataset loading script: %s',
                            name_or_path)
                self.datasets = hf_load_dataset(name_or_path, **kwargs)
            else:
                # to access private datasets from modelscope
                if access_token is not None:
                    from modelscope.hub.api import HubApi
                    HubApi().login(access_token)

                # only support some datasets with "adaseq" subset.
                msdataset = MsDataset.load(
                    name_or_path, **kwargs, subset_name='adaseq')
                self.datasets = {k: v._hf_ds for k, v in msdataset.items()}

        else:
            raise RuntimeError('Unsupported dataset!')

        if self.valid is None and self.test is not None:
            self.datasets['valid'] = self.datasets['test']

    @property
    def train(self):
        return self.datasets.get('train', None)

    @property
    def dev(self):
        return self.datasets.get('valid', None)

    @property
    def valid(self):
        return self.datasets.get('valid', None)

    @property
    def test(self):
        return self.datasets.get('test', None)
