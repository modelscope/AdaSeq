# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from modelscope.utils.registry import Registry, build_from_cfg, default_group
from transformers import PreTrainedTokenizerBase

from uner.metainfo import DataCollators
from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.multi_label_span_typing_data_collator)
@dataclass
class MultiLabelSpanTypingDataCollatorWithPadding(DataCollatorWithPadding):
    """Padding method for multilabel span typing dataset."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)
        self.keep_fields.append('spans')

    def padding(self, batch: Dict[str, Any], fields: List[str], batch_size: int, max_length: int,
                padding_side: str) -> Dict[str, Any]:
        """Padding a batch. In addition to the fields padded by base class DataCollatorWithPadding,
           'mention_boundary'、'type_ids'、'mention_msk' are padded here.
        """

        max_span_count = max([len(x[0]) for x in batch['mention_boundary']])
        for i in range(batch_size):
            difference = max_span_count - len(batch['mention_boundary'][i][0])
            if difference > 0:
                batch['mention_boundary'][i][0] = batch['mention_boundary'][i][0] + [0] * difference
                batch['mention_boundary'][i][1] = batch['mention_boundary'][i][1] + [0] * difference
                batch['type_ids'][i] = batch['type_ids'][i] + ([[0] * len(batch['type_ids'][i][0])]) * difference
                batch['mention_msk'][i] = batch['mention_msk'][i] + [0] * difference
        return batch
