import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from modelscope.models.builder import MODELS

from uner.data.constant import PAD_LABEL_ID
from uner.metainfo import Models
from uner.models.base import Model
from uner.modules.decoders import CRF, PartialCRF
from uner.modules.dropouts import WordDropout
from uner.modules.encoders import Encoder


@MODELS.register_module(module_name=Models.sequence_labeling_model)
class SequenceLabelingModel(Model):

    def __init__(self,
                 num_labels: int,
                 encoder: Union[Encoder, str] = None,
                 word_dropout: Optional[float] = 0.0,
                 use_crf: Optional[bool] = True,
                 **kwargs):
        super(SequenceLabelingModel, self).__init__()
        self.num_labels = num_labels
        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            self.encoder = Encoder.from_config(
                cfg_dict_or_path=encoder, **kwargs)
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_labels)

        self.use_dropout = word_dropout > 0.0
        if self.use_dropout:
            self.dropout = WordDropout(word_dropout)

        self.use_crf = use_crf
        if use_crf:
            if kwargs.get('partial', False):
                self.crf = PartialCRF(num_labels, batch_first=True)
            else:
                self.crf = CRF(num_labels, batch_first=True)
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=PAD_LABEL_ID)

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embed = self.encoder(
            inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]

        if self.use_dropout:
            embed = self.dropout(embed)

        logits = self.linear(embed)

        if 'emission_mask' in inputs:
            mask = inputs['emission_mask']
            masked_lengths = mask.sum(-1).long()
            masked_logits = torch.zeros_like(logits)
            for i in range(len(mask)):
                masked_logits[
                    i, :masked_lengths[i], :] = logits[i].masked_select(
                        mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
            logits = masked_logits

        return {'logits': logits}

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self._forward(inputs)

        logits = outputs['logits']
        label_ids = inputs['label_ids']
        seq_lens = inputs['emission_mask'].sum(-1).long()
        mask = torch.arange(
            inputs['emission_mask'].shape[1],
            device=seq_lens.device)[None, :] < seq_lens[:, None]

        if self.training:
            loss = self._calculate_loss(logits, label_ids, mask)
            outputs = {'logits': logits, 'loss': loss}
        else:
            predicts = self.decode(logits, mask)
            outputs = {'logits': logits, 'predicts': predicts}

        return outputs

    def _calculate_loss(self, logits, targets, mask):
        if self.use_crf:
            targets = targets * mask
            loss = -self.crf(logits, targets, reduction='mean', mask=mask)
        else:
            loss = self.loss_fn(logits.transpose(1, 2), targets)
        return loss

    def decode(self, logits, mask):
        if self.use_crf:
            predicts = self.crf.decode(logits, mask=mask).squeeze(0)
        else:
            predicts = logits.argmax(-1)
        return predicts
