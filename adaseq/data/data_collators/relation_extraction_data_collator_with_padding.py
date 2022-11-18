# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import DataCollators

from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.relation_extraction_data_collator)
@dataclass
class RelationExtractionDataCollatorWithPadding(DataCollatorWithPadding):
    """Relation Extraction collator."""

    pad_label_id: int = PAD_LABEL_ID
