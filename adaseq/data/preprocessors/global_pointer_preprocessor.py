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

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, return_offsets=True, **kwargs)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for Global Pointer model."""

        output = super().__call__(data)
        # origin sequence length
        length = len(output['tokens']['offsets']) - 2 * int(self.add_special_tokens)

        # calculate span matrix
        label_matrix = np.zeros([len(self.label_to_id), length, length])
        spans = data['spans']
        for span in spans:
            type_id = self.label_to_id[span['type']]
            label_matrix[type_id][span['start']][span['end'] - 1] = 1
        output['label_matrix'] = label_matrix
        return output
