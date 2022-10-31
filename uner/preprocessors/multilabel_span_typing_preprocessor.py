from typing import Any, Dict, List, Union

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS

from uner.metainfo import Preprocessors
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(
    module_name=Preprocessors.multilabel_span_typing_preprocessor)
class MultiLabelSpanTypingPreprocessor(NLPPreprocessor):

    def __init__(self, model_dir: str, label2id, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.label2id = label2id

    # label_boundary:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[s,e]]
    # label_ids:
    #    in: [{'start': s, 'end': e, 'types': l}]
    #    out: [[0,1,0]] # [[*] * num_classes(one-hot type vector)]
    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        output = super().__call__(data)
        token_span_mapping = output['reverse_offset_mapping']
        mention_type_ids = []
        boundary_starts = []
        boundary_ends = []
        mention_mask = []
        for span in data['spans']:
            if span['start'] > len(token_span_mapping) or span[
                    'end'] + 1 > len(token_span_mapping):
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
        output[
            'mention_msk'] = mention_mask  # msk, 为了避开nlp_preprocessor对他做padding.
        output['type_ids'] = mention_type_ids
        output['spans'] = data['spans']
        return output
