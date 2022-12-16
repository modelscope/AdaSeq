# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS

from adaseq.data.constant import NON_ENTITY_LABEL, PARTIAL_LABEL, PARTIAL_LABEL_ID
from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor

logger = logging.getLogger(__name__)


@PREPROCESSORS.register_module(module_name=Preprocessors.sequence_labeling_preprocessor)
class SequenceLabelingPreprocessor(NLPPreprocessor):
    """Preprocessor for Sequence Labeling"""

    def __init__(
        self, model_dir: str, labels: List[str], tag_scheme: str = 'BIOES', **kwargs
    ) -> None:
        self.tag_scheme = tag_scheme.upper()
        if not self._is_valid_tag_scheme(self.tag_scheme):
            raise ValueError('Invalid tag scheme! Options: [BIO, BIOES]')
        # self.tag_scheme = self._determine_tag_scheme_from_labels(labels)
        label_to_id = self._gen_label_to_id_with_bio(labels, self.tag_scheme)
        logger.info('label_to_id: ' + str(label_to_id))
        super().__init__(model_dir, label_to_id=label_to_id, return_offsets=True, **kwargs)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for Sequence Labeling models."""
        output = super().__call__(data)
        if isinstance(data, Dict) and 'spans' in data:
            length = len(output['tokens']['mask']) - 2 * int(self.add_special_tokens)
            labels = self._spans_to_bio_labels(data['spans'], length, self.tag_scheme)
            output['label_ids'] = [
                PARTIAL_LABEL_ID if labels[i] == PARTIAL_LABEL else self.label_to_id[labels[i]]
                for i in range(length)
            ]
        return output

    @staticmethod
    def _is_valid_tag_scheme(tag_scheme: str):
        return tag_scheme in ['BIO', 'BIOES']

    @staticmethod
    def _determine_tag_scheme_from_labels(labels: List[str]) -> str:
        tag_scheme = 'BIO'
        for label in labels:
            if label[0] not in 'BIOES':
                raise ValueError(f'Unsupported label: {label}')
            if label[0] in 'ES':
                tag_scheme = 'BIOES'
        return tag_scheme

    @staticmethod
    def _gen_label_to_id_with_bio(labels: List[str], tag_scheme: str = 'BIOES') -> Dict[str, int]:
        label_to_id = {}
        if 'O' in tag_scheme:
            label_to_id['O'] = 0
        for label in labels:
            for tag in 'BIES':
                if tag in tag_scheme:
                    label_to_id[f'{tag}-{label}'] = len(label_to_id)
        return label_to_id

    @staticmethod
    def _spans_to_bio_labels(spans: List[Dict], length: int, tag_scheme: str = 'BIOES'):
        labels = [NON_ENTITY_LABEL] * length
        for span in spans:
            for i in range(span['start'], min(span['end'], length)):
                if span['type'] == PARTIAL_LABEL:
                    labels[i] = span['type']
                    continue
                if 'S' in tag_scheme and i == span['start'] == span['end'] - 1:
                    labels[i] = 'S-' + span['type']
                elif i == span['start']:
                    labels[i] = 'B-' + span['type']
                elif 'E' in tag_scheme and i == span['end'] - 1:
                    labels[i] = 'E-' + span['type']
                else:
                    labels[i] = 'I-' + span['type']
        return labels
