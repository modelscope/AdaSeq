# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from modelscope.models.builder import MODELS
from torch.nn import BCELoss

from uner.metainfo import Models
from uner.models.base import Model
from uner.modules.dropouts import WordDropout
from uner.modules.encoders import Encoder, SpanEncoder


@MODELS.register_module(module_name=Models.multilabel_span_typing_model)
class MultiLabelSpanTypingModel(Model):
    """Multi type span typing model using BCE loss."""

    def __init__(self,
                 num_labels: int,
                 encoder: Union[Encoder, str] = None,
                 span_encoder_method: str = 'concat',
                 word_dropout: Optional[float] = 0.0,
                 loss_function: str = 'BCE',
                 class_threshold: float = 0.5,
                 use_biaffine: bool = False,
                 **kwargs):
        super(MultiLabelSpanTypingModel, self).__init__()
        self.num_labels = num_labels
        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            self.encoder = Encoder.from_config(
                cfg_dict_or_path=encoder, **kwargs)

        self.span_encoder = SpanEncoder(self.encoder.config.hidden_size,
                                        span_encoder_method, **kwargs)
        self.linear_input_dim = self.span_encoder.output_dim

        self.num_labels = num_labels
        self.linear = nn.Linear(self.linear_input_dim, num_labels)

        self.use_dropout = word_dropout > 0.0
        if self.use_dropout:
            self.dropout = WordDropout(word_dropout)

        self.loss_function_type = loss_function
        self.class_threshold = class_threshold
        self.use_biaffine = use_biaffine

        if self.loss_function_type == 'BCE':
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = BCELoss()

    def token2span_encoder(self, inputs, **kwargs):
        embed = self.encoder(
            inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        # embed: B x W x K
        if self.use_dropout:
            embed = self.dropout(embed)
        # span_reprs: B x S x D: trans sentence sequence to mentions sequence
        # the shape of boundary_ids is batch_size x 2 x max_mention_num, providing the detected mentions
        span_boundary = inputs['mention_boundary']
        span_reprs = self.span_encoder(embed, span_boundary)
        return span_reprs

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # B*M x K
        if self.use_biaffine:
            span_start_reprs, span_end_reprs = self.token2span_encoder(inputs)
            span_reprs = span_start_reprs * span_end_reprs
        else:
            span_reprs = self.token2span_encoder(inputs)

        # B*M x label_num
        logits = self.linear(span_reprs)
        return {'logits': logits}

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # B x M -> B*M x 1
        outputs = self._forward(inputs)
        mask = inputs['mention_msk'].reshape(-1).unsqueeze(-1)
        if self.training:
            loss = self._calculate_loss(outputs['logits'], inputs['type_ids'],
                                        mask)
            outputs = {'logits': outputs['logits'], 'loss': loss}
        else:
            batch_size, max_mention_per_sent = inputs['type_ids'].shape[0:2]
            logits = outputs['logits'].reshape(batch_size,
                                               max_mention_per_sent, -1)
            predicts = self.classify(logits, inputs['mention_boundary'], mask)
            outputs = {'logits': outputs['logits'], 'predicts': predicts}
        return outputs

    def _calculate_loss(self, logits, targets, mask):
        logits = logits * mask  # B*M x L
        if self.loss_function_type == 'BCE':
            logits = self.sigmoid(logits)  # B*M x L
            targets = targets.reshape(-1, self.num_labels).to(
                torch.float32)  # B*M x L
            loss = self.loss_fn(logits, targets) * mask
        loss = loss.sum() / (mask.sum())
        return loss

    def classify(self, logits, mention_boundary, mask):
        if self.loss_function_type == 'BCE':
            logits = self.sigmoid(logits)  # B*M x L
            predicts = torch.where(logits > self.class_threshold, 1,
                                   0)  # B*M x L
        predicts = predicts.detach().cpu().numpy()
        batch_mention_boundaries = mention_boundary.detach().cpu().numpy()
        batch_mentions = []
        for sent_idx, sent_predcit in enumerate(predicts):
            sent_predict = predicts[sent_idx]
            sent_mention_boundary = batch_mention_boundaries[sent_idx]
            sent_mentions = []
            for mention_idx, mention_pred in enumerate(sent_predict):
                types = [i for i, p in enumerate(mention_pred) if p == 1]
                sent_mentions.append(
                    (sent_mention_boundary[0][mention_idx],
                     sent_mention_boundary[1][mention_idx], types))
            batch_mentions.append(sent_mentions)

        return batch_mentions
