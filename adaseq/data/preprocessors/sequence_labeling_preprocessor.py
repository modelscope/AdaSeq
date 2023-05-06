# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from typing import Any, Dict, List, Optional, Union

from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields

from adaseq.data.constant import NON_ENTITY_LABEL, PARTIAL_LABEL, PARTIAL_LABEL_ID
from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor

logger = logging.getLogger(__name__)


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.sequence_labeling_preprocessor)
class SequenceLabelingPreprocessor(NLPPreprocessor):
    """Preprocessor for Sequence Labeling"""

    def __init__(
        self,
        model_dir: str,
        labels: Optional[List[str]] = None,
        tag_scheme: str = 'BIOES',
        label_to_id: Optional[Dict[str, int]] = None,
        chunk_size: Optional[int] = None,
        chunk_num: Optional[int] = None,
        **kwargs,
    ) -> None:
        assert (
            labels is not None or label_to_id is not None
        ), 'Either `labels` or `label_to_id` must be set for `SequenceLabelingPreprocessor`'
        self.tag_scheme = tag_scheme.upper()
        if not self._is_valid_tag_scheme(self.tag_scheme):
            raise ValueError('Invalid tag scheme! Options: [BIO, BIOES]')
        # self.tag_scheme = self._determine_tag_scheme_from_labels(labels)
        if label_to_id is None:
            label_to_id = self._gen_label_to_id_with_bio(labels, self.tag_scheme)
        logger.info('label_to_id: ' + str(label_to_id))
        super().__init__(model_dir, label_to_id=label_to_id, return_offsets=True, **kwargs)

        if not chunk_size:
            self.chunk_size = None
        else:
            self.chunk_size = chunk_size
            self.chunk_num = chunk_num
            self.max_length = kwargs.get('max_length')

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for Sequence Labeling models."""
        output = super().__call__(data)
        if not self.chunk_size:
            if isinstance(data, Dict) and 'spans' in data:
                length = len(output['tokens']['mask']) - 2 * int(self.add_special_tokens)
                labels = self._spans_to_bio_labels(data['spans'], length, self.tag_scheme)
                output['label_ids'] = [
                    PARTIAL_LABEL_ID if labels[i] == PARTIAL_LABEL else self.label_to_id[labels[i]]
                    for i in range(length)
                ]
        else:
            tokenizer = self.tokenizer
            tokens = output['tokens']
            # trunc
            length = sum(data['mask'])
            for i in range(1, length + 1):
                if tokens['offsets'][i][1] > self.chunk_size - 2 * int(self.add_special_tokens):
                    length = i - 1
                    break
            eos = tokens['offsets'][length + 1][0]
            orig_ids = tokens['input_ids'][1:eos]
            extra_ids = tokens['input_ids'][eos:-1]
            concat_size = self.chunk_size - 2 - len(orig_ids)
            # assert len(extra_ids) > concat_size * self.chunk_num and concat_size > 0
            ctx_ids = []
            ctx_attn_mask = []
            ctx_offsets = []
            for chunk_id in range(0, self.chunk_num):
                if concat_size > 0 and (chunk_id + 1) * concat_size <= len(extra_ids):
                    context = extra_ids[chunk_id * concat_size : (chunk_id + 1) * concat_size]
                    context = (
                        [tokenizer.cls_token_id] + orig_ids + [tokenizer.sep_token_id] + context
                    )
                    assert len(context) == self.chunk_size
                    attn_mask = [True] * len(context)
                    offset = tokens['offsets'][: length + 1]
                    offset = offset + [(0, 0)] * (len(context) - len(offset))
                else:
                    context = [tokenizer.cls_token_id] + orig_ids + [tokenizer.sep_token_id]
                    attn_mask = [True] * len(context)
                    context = context + [tokenizer.pad_token_id] * (self.chunk_size - len(context))
                    attn_mask = attn_mask + [False] * (self.chunk_size - len(attn_mask))
                    assert len(context) == len(attn_mask) == self.chunk_size
                    offset = tokens['offsets'][: length + 1]
                    offset = offset + [(0, 0)] * (len(context) - len(offset))
                ctx_ids.append(context)
                ctx_attn_mask.append(attn_mask)
                ctx_offsets.append(offset)
            tokens['input_ids'] = ctx_ids
            tokens['attention_mask'] = ctx_attn_mask
            tokens['offsets'] = ctx_offsets
            tokens['mask'] = [True] * length

            if isinstance(data, Dict) and 'spans' in data:
                labels = self._spans_to_bio_labels(data['spans'], length, self.tag_scheme)
                output['label_ids'] = [
                    PARTIAL_LABEL_ID if labels[i] == PARTIAL_LABEL else self.label_to_id[labels[i]]
                    for i in range(length)
                ]

        return output

    @staticmethod
    def _is_valid_tag_scheme(tag_scheme: str):
        return tag_scheme in ['BIO', 'BIOES', 'BI', 'BIES']

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
