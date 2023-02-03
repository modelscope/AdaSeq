# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import pickle

import datasets
import numpy as np
from datasets import DownloadManager, Features, Value

from .base import CustomDatasetBuilder


class MCCEDatasetBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for entity typing datasets"""

    def __init__(self, data_dir=None, data_files=None, **corpus_config):
        super().__init__(data_dir=data_dir, data_files=data_files)
        self.corpus_config = corpus_config


class MCCEDatasetBuilder(CustomDatasetBuilder):
    """Builder for entity typing datasets.

    features:
        id: string, data record id.
        tokens: list[str] input tokens.
        spans: List[Dict],  mentions like:
        [{'start': 0, 'end': 2, 'type': ['PER', 'MAN'], 'candidates': ['PER', 'MAN', 'ORG']}]
        mask: bool, mention mask.
    """

    BUILDER_CONFIG_CLASS = MCCEDatasetBuilderConfig

    def stub():  # noqa: D102
        pass

    def _info(self):
        info = datasets.DatasetInfo(
            features=Features(
                {
                    'id': Value('string'),
                    'tokens': [Value('string')],
                    'spans': [
                        {
                            'start': Value('int32'),  # close
                            'end': Value('int32'),  # open
                            'type': [Value('string')],
                            'candidates': [Value('string')],
                        }
                    ],
                    'mask': [Value('bool')],
                }
            )
        )
        return info

    def _split_generators(self, dl_manager: DownloadManager):
        """Specify feature dictionary generators and dataset splits.

        This function returns a list of `SplitGenerator`s defining how to generate
        data and what splits to use.

        Example:
            the gen_kwargs should contain the typing dataset filename and the corresponding candidates
            return[
                    datasets.SplitGenerator(
                            name=datasets.Split.TRAIN,
                            gen_kwargs={'file': 'train_data.zip', 'cand': List[List[str]]},
                    ),
                    datasets.SplitGenerator(
                            name=datasets.Split.TEST,
                            gen_kwargs={'file': 'test_data.zip', 'cand': List[List[str]]},
                    ),
            ]

        The above code will first call `_generate_examples(file='train_data.zip', cand=...)`
        to write the train data, then `_generate_examples(file='test_data.zip', cand=...)` to
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
            data should additionally contain a candidate file:
            {'train': List[List[str]], 'dev': List[List[str]], 'test': List[List[str]]},
            in which containing the candidates of each sample in the training/dev/test set.

        Returns:
            `list<SplitGenerator>`.
        """

        data_files = self._resolve_datasets(dl_manager=dl_manager)

        if 'cand' not in data_files:
            return [
                datasets.SplitGenerator(
                    name=split_name, gen_kwargs={'filepath': data_files[split_name], 'cand': None}
                )
                for split_name in data_files.keys()
            ]
        else:
            candidates = data_files.pop('cand')
            candidates = pickle.load(open(candidates, 'rb'))
            keys = candidates.keys()
            split_names = data_files.keys()
            for i in split_names:
                if i == 'valid' and i not in keys:
                    assert 'dev' in keys
                    candidates[i] = candidates['dev']
                elif i == 'dev' and i not in keys:
                    assert 'valid' in keys
                    candidates[i] = candidates['valid']
                else:
                    assert i in keys

            return [
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={'filepath': data_files[split_name], 'cand': candidates[split_name]},
                )
                for split_name in data_files.keys()
            ]

    def _generate_examples(self, filepath, cand):
        """Load data file.
        Args:
        file_path: string, data file path.
        corpus_config: dictionary, required keys:
        tokenizer: string, specify the tokenization method,
            'char' list(text) and 'blank' for text.split(' ')
        is_end_included: bool, wheather the end index pointer to the
            last token or the token next to the last token.
            e.g., text = 'This is an example.', mention = 'example',
            end = 3 if is_end_included is True.

        """
        corpus_config = self.config.corpus_config
        spans_key = corpus_config.get('spans_key', 'label')
        text_key = corpus_config.get('text_key', 'text')
        tokenizer = corpus_config.get('tokenizer', 'char')
        is_end_included = corpus_config.get('is_end_included', False)

        with open(filepath, encoding='utf-8') as f:
            guid = 0
            for line in f:
                example = json.loads(line)
                text = example[text_key]
                if isinstance(text, list):
                    tokens = text
                elif isinstance(text, str):
                    if tokenizer == 'char':
                        tokens = list(text)
                    elif tokenizer == 'blank':
                        tokens = text.split()
                    else:
                        raise NotImplementedError
                else:
                    raise ValueError('Unsupported text input.')

                entities = list()
                for span in example[spans_key]:
                    if is_end_included:
                        span['end'] += 1
                    entities.append(span)

                mask = [True] * len(tokens)

                reading_format = corpus_config.get('encoding_format', 'concat')
                if reading_format == 'concat':
                    for entity in entities:
                        if cand is None:
                            entity['candidates'] = None
                        else:
                            if type(cand[guid]) == np.ndarray:
                                candidates = cand[guid].tolist()
                            else:
                                candidates = cand[guid]
                            entity['candidates'] = candidates
                        yield guid, {
                            'id': str(guid),
                            'tokens': tokens,
                            'spans': [entity],
                            'mask': mask,
                        }
                        guid += 1
                else:
                    raise NotImplementedError('unimplemented reading_format')
