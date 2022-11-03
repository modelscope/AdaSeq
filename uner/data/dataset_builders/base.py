import os
from abc import ABC, abstractmethod

import datasets
from datasets import Features, Value


class CustomDatasetBuilder(ABC, datasets.GeneratorBasedBuilder):

    # datasets.load_dataset从builder script import_main_class时取得第一个非抽象类，
    # 所以需要把这个基类做成抽象类，否则load的是这个基类。（import 基类在前，继承类定义在后）
    @abstractmethod
    def stub():
        pass

    @classmethod
    def parse_label(cls, data):
        pass

    def _split_generators(self, dl_manager):
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
    for filename in files:
        if keyword in filename:
            return filename
    return None
