# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys

from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(
    group_key=Fields.nlp, module_name=Preprocessors.multilabel_span_typing_preprocessor
)
class MultiLabelSpanTypingPreprocessor(NLPPreprocessor):
    """Preprocessor for multilabel (aka multi-type) span typing task.
    span targets are processed into mention_boundary, type_ids.
    examples:
        span: {'start':1, 'end':2, 'type': ['PER']}
        processed: {'mention_boundary': [[1], [2]], 'type_ids':[1]}
    """

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, return_offsets=True, **kwargs)

    def __call__(self, data: Dict) -> Dict[str, Any]:
        """prepare inputs for Entity Typing model"""
        output = super().__call__(data)

        mention_type_ids = []
        boundary_starts = []
        boundary_ends = []
        mention_mask = []
        for span in data['spans']:
            boundary_starts.append(span['start'])
            boundary_ends.append(span['end'] - 1)
            mention_mask.append(1)
            if 'type' in span:
                type_ids = [self.label_to_id.get(x, -1) for x in span['type']]
                padded_type_ids = [0] * len(self.label_to_id)
                for t in type_ids:
                    if t != -1:
                        padded_type_ids[t] = 1
                mention_type_ids.append(padded_type_ids)

        output['mention_boundary'] = [boundary_starts, boundary_ends]
        output['mention_mask'] = mention_mask  # mask, 为了避开nlp_preprocessor对他做padding.
        if len(mention_type_ids) > 0:
            output['type_ids'] = mention_type_ids
        if self.mode == ModeKeys.INFERENCE:
            output['mention_boundary'] = np.expand_dims(np.array(output['mention_boundary']), 0)
            output['mention_mask'] = np.expand_dims(np.array(output['mention_mask']), 0)
        return output


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.multilabel_concat_typing_preprocessor
)
class MultiLabelConcatTypingPreprocessor(NLPPreprocessor):
    """Preprocessor for multilabel (aka multi-type) span concat typing task."""

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, return_offsets=True, **kwargs)

    # label_boundary:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[s,e]]
    # label_ids:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[0,1,0]] # [[*] * num_classes(one-hot type vector)]
    def __call__(self, data: Dict) -> Dict[str, Any]:  # noqa: D102
        spans = data.get('spans', [])
        assert len(spans) == 1, 'ConcatTyping only supports single mention per data'
        span = spans[0]
        tokens = data['tokens']
        type_ids = [self.label_to_id.get(x, -1) for x in span['type']]
        padded_type_ids = [0] * len(self.label_to_id)  # multilabel type id onehot vector
        for t in type_ids:
            if t != -1:
                padded_type_ids[t] = 1

        mention = tokens[span['start'] : span['end']]  # since end is open
        sent = (
            [self.tokenizer.cls_token]
            + tokens
            + [self.tokenizer.sep_token]
            + mention
            + [self.tokenizer.sep_token]
        )
        input_ids = []
        for tok in sent:
            subtoken_ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if len(subtoken_ids) == 0:
                subtoken_ids = [self.tokenizer.unk_token_id]
            input_ids.extend(subtoken_ids)

        if len(input_ids) > self.max_length:
            input_ids = [input_ids[0]] + input_ids[-(self.max_length - 1) :]
            # clip from the left except cls token

        attention_mask = [1] * len(input_ids)
        output = {
            'tokens': {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
            'type_ids': padded_type_ids,
            'meta': data,
        }
        return output


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.multilabel_concat_typing_mcce_preprocessor
)
class MultiLabelConcatTypingMCCEPreprocessor(NLPPreprocessor):
    """Preprocessor for multilabel (aka multi-type) span concat typing task.
    cand_size: number of candidates, -1 when use all labels candidates"""

    def __init__(self, model_dir: str, cand_size: int = -1, **kwargs):
        super().__init__(model_dir, return_offsets=True, **kwargs)
        self.cand_size = cand_size

    # label_boundary:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[s,e]]
    # label_ids:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[0,1,0]] # [[*] * num_classes(one-hot type vector)]
    def __call__(self, data: Dict) -> Dict[str, Any]:  # noqa: D102
        spans = data.get('spans', [])
        assert len(spans) == 1, 'ConcatTyping only supports single mention per data'
        span = spans[0]
        tokens = data['tokens']
        vocab_size = self.tokenizer.vocab_size
        if span['candidates'] is None:
            cands = np.arange(len(self.label_to_id))
        else:
            cands = np.array([self.label_to_id[i] for i in span['candidates']])

        if self.cand_size > 0:
            cands = cands[: self.cand_size]
        np.random.shuffle(cands)  # shuffle the candidates to avoid overfitting

        type_ids = [self.label_to_id[i] for i in span['type']]
        cand_ids = (cands + vocab_size).tolist()
        cand_labels = [1 if i in type_ids else 0 for i in cands]

        mention = tokens[span['start'] : span['end']]  # since end is open
        sent = (
            [self.tokenizer.cls_token]
            + tokens
            + [self.tokenizer.sep_token]
            + mention
            + [self.tokenizer.sep_token]
        )
        input_ids = []
        for tok in sent:
            subtoken_ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if len(subtoken_ids) == 0:
                subtoken_ids = [self.tokenizer.unk_token_id]
            input_ids.extend(subtoken_ids)

        assert self.max_length <= self.tokenizer.model_max_length
        # pad at right side
        if len(input_ids) + len(cand_ids) > self.max_length:
            # clip from the left except cls token
            input_ids += cand_ids
            input_ids = [input_ids[0]] + input_ids[-(self.max_length - 1) :]
            attention_mask = [1] * len(input_ids)

        else:
            remain = self.max_length - len(input_ids) - len(cand_ids)
            attention_mask = [1] * len(input_ids) + [0] * remain + [1] * len(cand_ids)
            input_ids += remain * [self.tokenizer.pad_token_id] + cand_ids

        assert len(input_ids) == self.max_length
        assert len(attention_mask) == self.max_length
        output = {
            'tokens': {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
            'type_ids': cand_labels,
            'cands': cands.tolist(),
            'meta': data,
        }
        return output
