# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from adaseq.metainfo import DataCollators

from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.span_extraction_data_collator)
class SpanExtractionDataCollatorWithPadding(DataCollatorWithPadding):
    """Padding method for span extraction dataset."""

    def padding(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Padding a batch. In addition to the fields padded by base class
        `DataCollatorWithPadding`, label_matrix is padded here.
        """
        field = 'span_labels'
        max_length = max(len(i[0]) for i in batch[field])
        for i in range(len(batch[field])):
            difference = max_length - len(batch[field][i][0])
            if difference > 0:
                padded_labels = np.zeros((max_length, max_length), dtype=int)
                padded_labels[: batch[field][i].shape[0], : batch[field][i].shape[1]] = batch[
                    field
                ][i].astype(int)
                batch[field][i] = padded_labels.tolist()
            else:
                batch[field][i] = batch[field][i].astype(int).tolist()

        return batch
