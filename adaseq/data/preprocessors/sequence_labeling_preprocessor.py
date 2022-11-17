# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS

from adaseq.data.constant import NON_ENTITY_LABEL, PARTIAL_LABEL, PARTIAL_LABEL_ID
from adaseq.metainfo import Preprocessors
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(module_name=Preprocessors.sequence_labeling_preprocessor)
class SequenceLabelingPreprocessor(NLPPreprocessor):
    """ Preprocessor for Sequence Labeling """

    def __init__(self, model_dir: str, labels: List[str] = None, tag_scheme: str = 'BIOES', **kwargs):
        super().__init__(model_dir, return_emission_mask=True, **kwargs)

        self.tag_scheme = tag_scheme.upper()
        if not self._is_valid_tag_scheme(self.tag_scheme):
            raise ValueError('Invalid tag scheme! Options: [BIO, BIOES]')

        label2id = kwargs.get('label2id', None)
        self.label2id = self.map_label_to_id(labels, label2id)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """ prepare inputs for Sequence Labeling models. """
        output = super().__call__(data)
        if self.label2id is not None and isinstance(data, Dict) and 'spans' in data:
            input_length = sum(output['emission_mask'])
            labels = self._spans_to_bio_labels(data['spans'], input_length, self.tag_scheme)
            output['label_ids'] = [
                PARTIAL_LABEL_ID if labels[i] == PARTIAL_LABEL else self.label2id[labels[i]]
                for i in range(input_length)
            ]
        return output

    def _label2id(self, labels: List[str]) -> Dict[str, int]:
        return self._gen_label2id_with_bio(labels, self.tag_scheme)

    @staticmethod
    def _is_valid_tag_scheme(tag_scheme: str):
        return tag_scheme in ['BIO', 'BIOES']

    @staticmethod
    def _gen_label2id_with_bio(labels: List[str], tag_scheme: str = 'BIOES') -> Dict[str, int]:
        label2id = {}
        if 'O' in tag_scheme:
            label2id['O'] = 0
        for label in labels:
            for tag in 'BIES':
                label2id[f'{tag}-{label}'] = len(label2id)
        return label2id

    @staticmethod
    def _spans_to_bio_labels(spans: List[Dict], length: int, tag_scheme: str = 'BIOES'):
        labels = [NON_ENTITY_LABEL] * length
        for span in spans:
            for i in range(span['start'], min(span['end'], length)):
                if span['type'] == PARTIAL_LABEL:
                    labels[i] = span['type']
                    continue
                if 'S' in tag_scheme and i == span['start'] == span['end'] + 1:
                    labels[i] = 'S-' + span['type']
                elif i == span['start']:
                    labels[i] = 'B-' + span['type']
                elif 'E' in tag_scheme and i == span['end'] + 1:
                    labels[i] = 'E-' + span['type']
                else:
                    labels[i] = 'I-' + span['type']
        return labels
