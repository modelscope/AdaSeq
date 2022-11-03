from typing import Any, Dict, List, Union

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS

from uner.data.constant import NON_ENTITY_LABEL, PAD_LABEL, PAD_LABEL_ID
from uner.metainfo import Preprocessors
from uner.utils.data_utils import gen_label2id
from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(
    module_name=Preprocessors.global_pointer_preprocessor)
class GlobalPointerPreprocessor(NLPPreprocessor):

    def __init__(self, model_dir: str, labels: List[str], *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        label2id = kwargs.get('label2id', None)
        self.label2id = self.map_label_to_id(labels, label2id)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        output = super().__call__(data)
        token_span_mapping = output['reverse_offset_mapping']

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

    def _label2id(self, labels: List[str]) -> Dict[str, int]:
        return gen_label2id(labels)
