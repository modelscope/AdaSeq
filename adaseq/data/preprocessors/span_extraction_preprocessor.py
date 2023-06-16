# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields

from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.span_extraction_preprocessor)
class SpanExtracionPreprocessor(NLPPreprocessor):
    """Preprocessor of span-based model.
    span targets are processed into `span_labels`
    """

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, return_offsets=True, **kwargs)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for span-based model."""

        output = super().__call__(data)

        if not (isinstance(data, dict) and 'spans' in data):
            return output

        # origin sequence length
        length = len(output['tokens']['mask']) - 2 * int(self.add_special_tokens)
        # calculate span labels
        span_labels = np.zeros([length, length])
        for span in data['spans']:
            # self.label_to_id doesn't have non-entity label,
            # we set index 0 as non-entity label, so we add 1 to type_id
            type_id = self.label_to_id[span['type']] + 1
            if span['end'] > length:
                continue
            span_labels[span['start']][span['end'] - 1] = type_id
        output['span_labels'] = span_labels
        return output
