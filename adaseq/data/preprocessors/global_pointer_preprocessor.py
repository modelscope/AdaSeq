# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS

from adaseq.metainfo import Preprocessors
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(module_name=Preprocessors.global_pointer_preprocessor)
class GlobalPointerPreprocessor(NLPPreprocessor):
    """Preprocessor of global pointer model.
    span targets are processed into label_matrixes
    """

    def __init__(self, model_dir: str, labels: List[str], **kwargs):
        super().__init__(model_dir, return_offsets_mapping=True, **kwargs)

        label2id = kwargs.get('label2id', None)
        self.label2id = self.map_label_to_id(labels, label2id)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """ prepare inputs for Global Pointer model. """

        output = super().__call__(data)

        token_span_mapping = []
        for i, (token_start, token_end) in enumerate(output['offset_mapping']):
            if token_start == token_end and token_start == 0:
                token_span_mapping.append([0, 0])  # CLS, SEP
            elif token_start == token_end:
                token_span_mapping[-1][1] += 1
            else:
                token_span_mapping.append([i, i + 1])

        # calculate span matrix，be careful to fix offset: 1, offset_mapping, 2, cls_token
        label_matrix = np.zeros([len(self.label2id), len(output['input_ids']), len(output['input_ids'])])
        spans = data['spans']
        for span in spans:
            if span['start'] > len(token_span_mapping) or span['end'] + 1 > len(token_span_mapping):
                continue
            start = token_span_mapping[span['start'] + 1][0]
            end = token_span_mapping[span['end']][1] - 1
            type_id = self.label2id[span['type']]
            label_matrix[type_id][start][end] = 1
        output['label_matrix'] = label_matrix
        output['spans'] = data['spans']
        return output
