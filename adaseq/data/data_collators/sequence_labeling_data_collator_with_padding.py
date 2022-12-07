# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import DataCollators

from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.sequence_labeling_data_collator)
class SequenceLabelingDataCollatorWithPadding(DataCollatorWithPadding):
    """Collator for the sequence labeling task"""

    pad_label_id: int = PAD_LABEL_ID

    def padding(self, batch: Dict[str, Any], padding_side: str, **kwargs) -> Dict[str, Any]:
        """pad label sequence `label_ids`"""
        return super().padding(batch, padding_side, 'label_ids', self.pad_label_id)
