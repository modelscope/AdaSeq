# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Set

from adaseq.metainfo import DataCollators

from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.pretraining_data_collator)
class PretrainingDataCollatorWithPadding(DataCollatorWithPadding):
    """Padding method for pretraining dataset."""

    def __init__(
        self, tokenizer, default_pad_id: int = 0, no_pad_fields: Optional[Set[str]] = None, **kwargs
    ) -> None:
        super().__init__(tokenizer, default_pad_id, no_pad_fields, **kwargs)
        self.keep_fields.add('prompt_type')
