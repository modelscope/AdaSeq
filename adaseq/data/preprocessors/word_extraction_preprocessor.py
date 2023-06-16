# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

import numpy as np
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields

from adaseq.metainfo import Preprocessors

from .nlp_preprocessor import NLPPreprocessor

dis2idx = np.zeros(1000, dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.word_extraction_preprocessor)
class WordExtracionPreprocessor(NLPPreprocessor):
    """Preprocessor of span-based model.
    span targets are processed into `span_labels`
    """

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, return_offsets=True, **kwargs)

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """prepare inputs for w2ner model."""
        output = super().__call__(data)
        if not (isinstance(data, dict) and 'spans' in data):
            return output

        # origin sequence length
        length = len(output['tokens']['mask']) - 2 * int(self.add_special_tokens)

        _grid_labels = np.zeros((length, length), dtype=np.int64)
        _pieces2word = np.zeros((length, len(output['tokens']['input_ids'])), dtype=np.bool_)
        _dist_inputs = np.zeros((length, length), dtype=np.int64)
        _grid_mask2d = np.ones((length, length), dtype=np.bool_)
        all_offsets = output['tokens']['offsets']

        for i in range(1, len(all_offsets) - 1):
            offset = output['tokens']['offsets'][i]
            _pieces2word[i - 1, offset[0] : offset[1] + 1] = 1

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for span in data['spans']:
            start, end = span['start'], span['end']
            if end > length:
                continue
            # self.label_to_id doesn't have non-entity label,
            # type_id needs to be added with 2, because another 2 types [PAD] and [SUK] is also included
            type_id = self.label_to_id[span['type']] + 2
            for i in range(start, end - 1):
                _grid_labels[i, i + 1] = 1
            assert end > start
            _grid_labels[end - 1, start] = type_id

        output['grid_labels'] = _grid_labels
        output['dist_inputs'] = _dist_inputs
        output['grid_mask2d'] = _grid_mask2d
        output['pieces2word'] = _pieces2word
        output['sent_length'] = length
        return output
