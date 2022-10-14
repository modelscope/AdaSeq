from typing import Any, Dict, List, Union

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS

from uner.metainfo import Preprocessors
from .constant import NON_ENTITY_LABEL, PAD_LABEL, PAD_LABEL_ID
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(
    module_name=Preprocessors.global_pointer_preprocessor)
class GlobalPointerPreprocessor(NLPPreprocessor):

    def __init__(self, model_dir: str, label2id, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.label2id = label2id

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        output = super().__call__(data)
        token_span_mapping = self.compress_token_mapping(
            output['offset_mapping'])

        # 计算span矩阵，注意修正offset: 1, offset_mapping, 2, cls_token
        label_matrix = np.zeros([
            len(self.label2id),
            len(output['input_ids']),
            len(output['input_ids'])
        ])
        spans = data['spans']
        for span in spans:
            if span['start'] > len(token_span_mapping) or span[
                    'end'] + 1 > len(token_span_mapping):
                continue
            start = token_span_mapping[span['start'] + 1][0]
            end = token_span_mapping[span['end']][1] - 1
            type_id = self.label2id[span['type']]
            label_matrix[type_id][start][end] = 1
        output['label_matrix'] = label_matrix
        output['spans'] = data['spans']
        return output
        '''
        offset_mapping
            tokens : 0, 1, 2
            subtoken: 0, 1-1, 1-2, 2-1, 2-2
            offset_mapping: [(0,1), (1,2), (2,2), (2,3), (3,3)]
        compressed:
            offset_mapping: [(0,1), (1,3), (3,5)]
        '''

    def compress_token_mapping(self, original_token_mapping):
        token_span_mapping = []
        for i, (token_start, token_end) in enumerate(original_token_mapping):
            if token_start == token_end and token_start == 0:
                token_span_mapping.append([0, 0])  # CLS, SEP
            elif token_start == token_end:
                token_span_mapping[-1][1] += 1
            else:
                token_span_mapping.append([i, i + 1])
        return token_span_mapping
