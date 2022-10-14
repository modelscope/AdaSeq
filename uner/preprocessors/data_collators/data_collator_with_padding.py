from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from transformers import PreTrainedTokenizerBase

from ..constant import PAD_LABEL_ID
from .base import DataBatch


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    pad_label_id: int = PAD_LABEL_ID

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(features)
        fields = features[0].keys()
        batch = {key: [example[key] for example in features] for key in fields}

        input_ids_field = 'input_ids' if 'ext_input_ids' not in fields else 'ext_input_ids'
        max_length = max(
            [len(input_ids) for input_ids in batch[input_ids_field]])
        padding_side = self.tokenizer.padding_side

        for i in range(batch_size):
            for field in fields:
                if field.endswith('matrix'):
                    difference = max_length - len(batch[field][i][0])
                else:
                    difference = max_length - len(batch[field][i])
                if difference > 0:
                    if field.endswith('input_ids'):
                        pad_id = self.tokenizer.pad_token_id
                    elif field.endswith('token_type_ids'):
                        pad_id = self.tokenizer.pad_token_type_id
                    elif field.endswith('label_ids'):
                        pad_id = self.pad_label_id
                    elif field.endswith('mask'):
                        pad_id = 0
                    elif field.endswith('matrix'):
                        pad_id = 0
                    else:
                        continue

                    if isinstance(batch[field][i], np.ndarray):
                        # label_matrix
                        num_classes = len(batch[field][i])
                        padded_label_matrix = np.ones(
                            (num_classes, max_length, max_length)) * pad_id
                        padded_label_matrix[:, :batch[field][i].
                                            shape[1], :batch[field][i].
                                            shape[2]] = batch[field][i]
                        batch[field][i] = padded_label_matrix
                    else:
                        if padding_side == 'right':
                            batch[field][i] = batch[field][i] + [pad_id
                                                                 ] * difference
                        elif padding_side == 'left':
                            batch[field][i] = [pad_id
                                               ] * difference + batch[field][i]
                        else:
                            raise ValueError('Invalid padding strategy:'
                                             + str(self.padding_side))

        batch = DataBatch(batch)
        return batch
