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
        field = 'label_matrix'
        max_length = max(len(i[0]) for i in batch[field])
        for i in range(len(batch[field])):
            difference = max_length - len(batch[field][i][0])
            if difference > 0:
                # label_matrix
                num_classes = len(batch[field][i])
                padded_label_matrix = np.zeros((num_classes, max_length, max_length))
                padded_label_matrix[
                    :, : batch[field][i].shape[1], : batch[field][i].shape[2]
                ] = batch[field][i]
                batch[field][i] = padded_label_matrix.tolist()

        return batch
