# Modifications Copyright 2022 Alibaba, Inc. and its affiliates.

# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to MSRA NER Dataset"""

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@inproceedings{levow2006third,
  author    = {Gina{-}Anne Levow},
  title     = {The Third International Chinese Language Processing Bakeoff: Word
               Segmentation and Named Entity Recognition},
  booktitle = {SIGHAN@COLING/ACL},
  pages     = {108--117},
  publisher = {Association for Computational Linguistics},
  year      = {2006}
}
"""

_DESCRIPTION = """\
The Third International Chinese Language
Processing Bakeoff was held in Spring
2006 to assess the state of the art in two
important tasks: word segmentation and
named entity recognition. Twenty-nine
groups submitted result sets in the two
tasks across two tracks and a total of five
corpora. We found strong results in both
tasks as well as continuing challenges.
MSRA NER is one of the provided dataset.
There are three types of NE, PER (person),
ORG (organization) and LOC (location).
The dataset is in the BIO scheme.
For more details see https://faculty.washington.edu/levow/papers/sighan06.pdf
"""

_URL = 'https://www.modelscope.cn/api/v1/datasets/izhx404/toy_msra/repo/files?Revision=master&FilePath='
_TRAINING_FILE = 'train.txt'
_TEST_FILE = 'test.txt'


class MsraNerConfig(datasets.BuilderConfig):
    """BuilderConfig for MsraNer"""

    def __init__(self, **kwargs):
        """BuilderConfig for MSRA NER.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MsraNerConfig, self).__init__(**kwargs)


class MsraNer(datasets.GeneratorBasedBuilder):
    """MSRA NER dataset."""

    BUILDER_CONFIGS = [
        MsraNerConfig(
            name='adaseq',
            version=datasets.Version('1.0.0'),
            description='MSRA NER dataset'),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'id':
                datasets.Value('string'),
                'tokens':
                datasets.Sequence(datasets.Value('string')),
                'spans': [{
                    'start': datasets.Value('int32'),  # close
                    'end': datasets.Value('int32'),  # open
                    'type': datasets.Value('string')
                }]
            }),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            'train': _URL + _TRAINING_FILE,
            'test': _URL + _TEST_FILE,
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={'filepath': downloaded_files['train']}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': downloaded_files['test']}),
        ]

    def _generate_examples(self, filepath):

        def output_pair():
            spans = self._labels_to_spans(ner_tags)
            return guid, {'id': str(guid), 'tokens': tokens, 'spans': spans}

        logger.info('⏳ Generating examples from = %s', filepath)
        with open(filepath, encoding='utf-8') as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                line_stripped = line.strip()
                if line_stripped == '':
                    if tokens:
                        yield output_pair()
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    splits = line_stripped.split('\t')
                    if len(splits) == 1:
                        splits.append('O')
                    tokens.append(splits[0])
                    ner_tags.append(splits[1])
            # last example
            yield output_pair()

    @classmethod
    def _labels_to_spans(cls, labels):
        spans = []
        in_entity = False
        start = -1
        for i in range(len(labels)):
            # fix label error
            if labels[i][0] in 'IE' and not in_entity:
                labels[i] = 'B' + labels[i][1:]
            if labels[i][0] in 'BS':
                if i + 1 < len(labels) and labels[i + 1][0] in 'IE':
                    start = i
                else:
                    spans.append({
                        'start': i,
                        'end': i + 1,
                        'type': labels[i][2:]
                    })
            elif labels[i][0] in 'IE':
                if i + 1 >= len(labels) or labels[i + 1][0] not in 'IE':
                    assert start >= 0, \
                        'Invalid label sequence found: {}'.format(labels)
                    spans.append({
                        'start': start,
                        'end': i + 1,
                        'type': labels[i][2:]
                    })
                    start = -1
            if labels[i][0] in 'B':
                in_entity = True
            elif labels[i][0] in 'OES':
                in_entity = False
        return spans
