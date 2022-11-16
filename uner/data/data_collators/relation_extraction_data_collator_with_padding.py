# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Any, Dict, List

from uner.data.constant import PAD_LABEL_ID
from uner.metainfo import DataCollators
from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.relation_extraction_data_collator)
@dataclass
class RelationExtractionDataCollatorWithPadding(DataCollatorWithPadding):
    """ Relation Extraction collator. """

    pad_label_id: int = PAD_LABEL_ID

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)

    def padding(self, batch: Dict[str, Any], fields: List[str], batch_size: int, max_length: int,
                padding_side: str) -> Dict[str, Any]:
        """ do noting """
        return batch
