# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from adaseq.metainfo import DataCollators

from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.multi_label_span_typing_data_collator)
class MultiLabelSpanTypingDataCollatorWithPadding(DataCollatorWithPadding):
    """Padding method for multilabel span typing dataset."""

    def padding(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Padding a batch. In addition to the fields padded by base class DataCollatorWithPadding,
        'mention_boundary'、'type_ids'、'mention_mask' are padded here.
        """

        max_span_count = max([len(x[0]) for x in batch['mention_boundary']])
        for type_ids in batch['type_ids']:
            if len(type_ids) > 0:
                type_num = len(type_ids[0])
                break
        for i in range(len(batch['mention_boundary'])):
            difference = max_span_count - len(batch['mention_boundary'][i][0])
            if difference > 0:
                batch['mention_boundary'][i][0] = batch['mention_boundary'][i][0] + [0] * difference
                batch['mention_boundary'][i][1] = batch['mention_boundary'][i][1] + [0] * difference
                batch['type_ids'][i] = batch['type_ids'][i] + ([[0] * type_num]) * difference
                batch['mention_mask'][i] = batch['mention_mask'][i] + [0] * difference
        return batch


@DATA_COLLATORS.register_module(module_name=DataCollators.multi_label_concat_typing_data_collator)
class MultiLabelConcatTypingDataCollatorWithPadding(DataCollatorWithPadding):
    """Padding method for multilabel span concat typing dataset."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.no_pad_fields.add('type_ids')
