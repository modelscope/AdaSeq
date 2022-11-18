# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

from modelscope.preprocessors.builder import PREPROCESSORS

from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(module_name=Preprocessors.multilabel_span_typing_preprocessor)
class MultiLabelSpanTypingPreprocessor(NLPPreprocessor):
    """Preprocessor for multilabel (aka multi-type) span typing task.
    span targets are processed into mention_boundary, type_ids.
    examples:
        span: {'start':1, 'end':2, 'type': ['PER']}
        processed: {'mention_boundary': [[1], [2]], 'type_ids':[1]}
    """

    def __init__(self, model_dir: str, labels: List[str], **kwargs):
        super().__init__(model_dir, return_offsets_mapping=True, **kwargs)

        label2id = kwargs.get('label2id', None)
        self.label2id = self.map_label_to_id(labels, label2id)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for Entity Typing model"""
        output = super().__call__(data)

        token_span_mapping = []
        for i, (token_start, token_end) in enumerate(output['offset_mapping']):
            if token_start == token_end and token_start == 0:
                token_span_mapping.append([0, 0])  # CLS, SEP
            elif token_start == token_end:
                token_span_mapping[-1][1] += 1
            else:
                token_span_mapping.append([i, i + 1])

        mention_type_ids = []
        boundary_starts = []
        boundary_ends = []
        mention_mask = []
        for span in data['spans']:
            if span['start'] > len(token_span_mapping) or span['end'] + 1 > len(token_span_mapping):
                continue
            start = token_span_mapping[span['start'] + 1][0]
            end = token_span_mapping[span['end']][1] - 1
            boundary_starts.append(start)
            boundary_ends.append(end)
            mention_mask.append(1)
            type_ids = [self.label2id.get(x, -1) for x in span['type']]
            padded_type_ids = [0] * len(self.label2id)
            for t in type_ids:
                if t != -1:
                    padded_type_ids[t] = 1
            mention_type_ids.append(padded_type_ids)

        output['mention_boundary'] = [boundary_starts, boundary_ends]
        output['mention_msk'] = mention_mask  # msk, 为了避开nlp_preprocessor对他做padding.
        output['type_ids'] = mention_type_ids
        output['spans'] = data['spans']
        return output


@PREPROCESSORS.register_module(module_name=Preprocessors.multilabel_concat_typing_preprocessor)
class MultiLabelConcatTypingPreprocessor(NLPPreprocessor):
    """Preprocessor for multilabel (aka multi-type) span concat typing task."""

    def __init__(self, model_dir: str, labels: List[str], **kwargs):
        super().__init__(model_dir, return_offsets_mapping=True, **kwargs)

        label2id = kwargs.get('label2id', None)
        self.label2id = self.map_label_to_id(labels, label2id)

    # label_boundary:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[s,e]]
    # label_ids:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[0,1,0]] # [[*] * num_classes(one-hot type vector)]
    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:  # noqa: D102
        spans = data.get('spans', [])
        assert len(spans) == 1, 'ConcatTyping only supports single mention per data'
        span = spans[0]
        tokens = data['tokens']
        type_ids = [self.label2id.get(x, -1) for x in span['type']]
        padded_type_ids = [0] * len(self.label2id)  # multilabel type id onehot vector
        for t in type_ids:
            if t != -1:
                padded_type_ids[t] = 1

        mention = tokens[span['start'] : span['end']]  # TODO +1?
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
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'type_ids': padded_type_ids,
            'spans': spans,
            'tokens': tokens,
        }

        return output
