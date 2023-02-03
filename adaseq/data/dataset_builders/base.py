# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from abc import ABC, abstractmethod

import datasets
from datasets import DownloadManager
from datasets.utils.file_utils import is_remote_url


class CustomDatasetBuilder(ABC, datasets.GeneratorBasedBuilder):
    """Base class for custumized dataset builder."""

    @abstractmethod
    def stub():
        """Useless stub function.

        The datasets.load_dataset import the builder class from path via import_main_class.
        According to the source code, import_main_class will return the first non-abstract class.
        So we should make this base class a abstract class by adding this stub function.
        Otherwise, this base class will be returned by import_main_class,simce it is imported
        before the definition of sub-classes.
        """

        pass

    def _resolve_datasets(self, dl_manager: DownloadManager):
        """
        Resolve datasets. From (remote) data_dir or data_files
        Args:
            dl_manager:

        Returns:
            data_files

        """
        if self.config.data_dir is not None:
            if is_remote_url(self.config.data_dir):
                _data_dir = dl_manager.download_and_extract(self.config.data_dir)
            elif not os.path.isdir(self.config.data_dir):
                # could be some archieve files like `DIR/FILE.zip`
                _data_dir = dl_manager.extract(self.config.data_dir)
            else:
                _data_dir = self.config.data_dir

            data_files = dict()
            _all_files = os.listdir(_data_dir)
            for split_name in ['train', 'valid', 'test']:
                _file = get_file_by_keyword(_all_files, split_name)
                if _file is None and split_name == 'valid':
                    _file = get_file_by_keyword(_all_files, 'dev')
                if _file is None:
                    continue
                data_files[split_name] = os.path.join(_data_dir, _file)

        elif self.config.data_files is not None:
            assert isinstance(self.config.data_files, dict)

            data_files = dict()
            for k, v in self.config.data_files.items():
                if isinstance(v, list):
                    v = v[0]
                if not isinstance(v, str):
                    v = v.as_posix()
                if is_remote_url(v):
                    v = dl_manager.download(v)
                data_files[k] = v
        else:
            raise ValueError('Datasets cannot be resolved!')

        return data_files

    def _split_generators(self, dl_manager: DownloadManager):
        """Specify feature dictionary generators and dataset splits.

        This function returns a list of `SplitGenerator`s defining how to generate
        data and what splits to use.

        Example:

            return[
                    datasets.SplitGenerator(
                            name=datasets.Split.TRAIN,
                            gen_kwargs={'file': 'train_data.zip'},
                    ),
                    datasets.SplitGenerator(
                            name=datasets.Split.TEST,
                            gen_kwargs={'file': 'test_data.zip'},
                    ),
            ]

        The above code will first call `_generate_examples(file='train_data.zip')`
        to write the train data, then `_generate_examples(file='test_data.zip')` to
        write the test data.

        Datasets are typically split into different subsets to be used at various
        stages of training and evaluation.

        Note that for datasets without a `VALIDATION` split, you can use a
        fraction of the `TRAIN` data for evaluation as you iterate on your model
        so as not to overfit to the `TEST` data.

        For downloads and extractions, use the given `download_manager`.
        Note that the `DownloadManager` caches downloads, so it is fine to have each
        generator attempt to download the source data.

        A good practice is to download all data in this function, and then
        distribute the relevant parts to each split with the `gen_kwargs` argument

        Args:
            dl_manager: (DownloadManager) Download manager to download the data

        Returns:
            `list<SplitGenerator>`.
        """

        data_files = self._resolve_datasets(dl_manager=dl_manager)

        return [
            datasets.SplitGenerator(
                name=split_name, gen_kwargs={'filepath': data_files[split_name]}
            )
            for split_name in data_files.keys()
        ]


def get_file_by_keyword(files, keyword):
    """Get file by keyword, such as: train/test/dev."""

    for filename in files:
        if keyword in filename:
            return filename
    return None
