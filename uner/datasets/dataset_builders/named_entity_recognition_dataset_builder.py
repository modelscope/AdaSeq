import os

import datasets
from datasets import Features, Value

from uner.datasets.dataset_builders.dataset_reader import \
    NamedEntityRecognitionDatasetReader  # yapf: disable


class NamedEntityRecognitionDatasetBuilderConfig(datasets.BuilderConfig):

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super(NamedEntityRecognitionDatasetBuilderConfig, self).__init__(
            data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class NamedEntityRecognitionDatasetBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = NamedEntityRecognitionDatasetBuilderConfig

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features({
                'id':
                Value('string'),
                'tokens': [Value('string')],
                'spans': [{
                    'start': Value('int32'),  # close
                    'end': Value('int32'),  # open
                    'type': Value('string')
                }],
                'mask': [Value('bool')]
            }))
        return info

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

    def _generate_examples(self, filepath):
        if 'corpus_reader' in self.config.corpus_config:
            # TODO: get the reder via reflection
            raise NotImplementedError
        else:
            return NamedEntityRecognitionDatasetReader.load_data_file(
                filepath, self.config.corpus_config)


def get_file_by_keyword(files, keyword):
    for filename in files:
        if keyword in filename:
            return filename
    return None
