# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from modelscope.utils.registry import Registry, build_from_cfg, default_group
from transformers import PreTrainedTokenizerBase

from uner.data.constant import PAD_LABEL_ID
from uner.metainfo import DataCollators
from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.sequence_labeling_data_collator)
@dataclass
class SequenceLabelingDataCollatorWithPadding(DataCollatorWithPadding):
    pad_label_id: int = PAD_LABEL_ID

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)

    def padding(self, batch: Dict[str, Any], fields: List[str], batch_size: int, max_length: int,
                padding_side: str) -> Dict[str, Any]:
        for i in range(batch_size):
            field = 'label_ids'
            difference = max_length - len(batch[field][i])
            if difference > 0:
                pad_id = self.pad_label_id

                if padding_side == 'right':
                    batch[field][i] = batch[field][i] + [pad_id] * difference
                elif padding_side == 'left':
                    batch[field][i] = [pad_id] * difference + batch[field][i]
                else:
                    raise ValueError('Invalid padding strategy:' + str(self.padding_side))

        return batch
