# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from adaseq.metainfo import DataCollators

from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.twostage_data_collator)
class TwostageDataCollatorWithPadding(DataCollatorWithPadding):
    """Padding method for two stage ner dataset."""

    def padding(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Padding a batch. In addition to the fields padded by base class DataCollatorWithPadding,
        'mention_boundary', 'type_ids', 'mention_mask', 'ident_ids', 'ident_mask' are padded here.
        """

        max_length = max(len(i) for i in batch['ident_ids'])
        for i in range(len(batch['ident_ids'])):
            difference = max_length - len(batch['ident_ids'][i])
            if difference > 0:
                batch['ident_ids'][i] = batch['ident_ids'][i] + [-100] * difference

        max_span_count = max([len(x[0]) for x in batch['mention_boundary']])
        max_span_count = max(max_span_count, 1)
        for i in range(len(batch['mention_boundary'])):
            difference = max_span_count - len(batch['mention_boundary'][i][0])
            if difference > 0:
                batch['mention_boundary'][i][0] = batch['mention_boundary'][i][0] + [0] * difference
                batch['mention_boundary'][i][1] = batch['mention_boundary'][i][1] + [0] * difference
                batch['type_ids'][i] = batch['type_ids'][i] + [-100] * difference
                batch['mention_mask'][i] = batch['mention_mask'][i] + [0] * difference
        return batch
