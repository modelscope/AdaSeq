import os
from abc import ABC, abstractmethod
from typing import Set

import datasets
from datasets import Features, Value
from torch.utils.data import Dataset


class CustomDatasetBuilder(ABC, datasets.GeneratorBasedBuilder):
    """ Base class for custumized dataset builder."""

    @abstractmethod
    def stub():
        """ Useless stub function.

       The datasets.load_dataset import the builder class from path via import_main_class.
       According to the source code, import_main_class will return the first non-abstract class.
       So we should make this base class a abstract class by adding this stub function.
       Otherwise, this base class will be returned by import_main_class,simce it is imported
       before the definition of sub-classes.
       """

        pass

    @classmethod
    def parse_label(cls, data: Dataset) -> Set[str]:
        """Collect type labels from dataset."""

        pass

    def _split_generators(self, dl_manager):
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

        if self.config.data_files is not None:
            data_files = dl_manager.download_and_extract(
                self.config.data_files)
            if isinstance(data_files, dict):
                return [
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={'filepath': data_files[split_name][0]})
                    for split_name in data_files.keys()
                ]
            elif isinstance(data_files, (str, list, tuple)):
                if isinstance(data_files, str):
                    data_files = [data_files]
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={'filepath': data_files[0]})
                ]
        elif self.config.data_dir is not None:
            data_dir = dl_manager.download_and_extract(self.config.data_dir)
            all_files = os.listdir(data_dir)
            splits = []
            for split_name in ['train', 'valid', 'test']:
                data_file = get_file_by_keyword(all_files, split_name)
                if data_file is None and split_name == 'valid':
                    data_file = get_file_by_keyword(all_files, 'dev')
                if data_file is None:
                    continue
                data_file = os.path.join(data_dir, data_file)
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name, gen_kwargs={'filepath': data_file}))
            return splits
        else:
            raise ValueError('Datasets cannot be resolved!')


def get_file_by_keyword(files, keyword):
    """Get file by keyword, such as: train/test/dev."""

    for filename in files:
        if keyword in filename:
            return filename
    return None
