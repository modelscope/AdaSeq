# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
import torch

from adaseq.metainfo import DataCollators

from .base import DATA_COLLATORS, DataCollatorWithPadding


@DATA_COLLATORS.register_module(module_name=DataCollators.word_extraction_data_collator)
class WordExtractionDataCollatorWithPadding(DataCollatorWithPadding):
    """Padding method for span extraction dataset."""

    def padding(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Padding a batch. In addition to the fields padded by base class
        `DataCollatorWithPadding`, label_matrix is padded here.
        """
        pieces2word = batch['pieces2word']
        max_length = np.max([x.shape[0] for x in pieces2word])
        max_pie = np.max([x.shape[1] for x in pieces2word])
        fields = {
            'grid_labels': (1, 1),
            'dist_inputs': (1, 1),
            'grid_mask2d': (1, 0),
            'pieces2word': (0, 0),
        }
        for field in fields:
            if fields[field][0]:
                cur_part = batch[field]
                len1, len2 = max_length, max_length
            else:
                cur_part = batch[field]
                len1, len2 = max_length, max_pie
            d_type = int if fields[field][1] else bool
            for i in range(len(batch[field])):
                difference = len1 + len2 - cur_part[i].shape[0] - cur_part[i].shape[1]
                if difference > 0:
                    padded_labels = np.zeros((len1, len2), dtype=d_type)
                    padded_labels[: cur_part[i].shape[0], : cur_part[i].shape[1]] = cur_part[
                        i
                    ].astype(d_type)
                    cur_part[i] = padded_labels.tolist()
                else:
                    cur_part[i] = cur_part[i].astype(d_type).tolist()

        field = 'sent_length'
        batch[field] = torch.LongTensor(batch[field])
        pieces_name = 'pieces2word'
        batch['tokens'][pieces_name] = batch[pieces_name]
        batch.pop(pieces_name)
        return batch
