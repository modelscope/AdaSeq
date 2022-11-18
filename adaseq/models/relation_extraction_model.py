# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models.builder import MODELS
from torch.nn import LayerNorm

from adaseq.metainfo import Models
from adaseq.models.base import Model
from adaseq.modules.dropouts import WordDropout
from adaseq.modules.encoders import Encoder


@MODELS.register_module(module_name=Models.relation_extraction_model)
class RelationExtractionModel(Model):
    """Relation extraction model

    This model is used for relation extraction tasks.
    """

    def __init__(
        self,
        num_labels: int,
        encoder: Union[Encoder, str] = None,
        word_dropout: Optional[float] = 0.0,
        multiview: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        **kwargs
    ):
        super(RelationExtractionModel, self).__init__()
        self.num_labels = num_labels
        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            self.encoder = Encoder.from_config(cfg_dict_or_path=encoder, **kwargs)

        self.use_dropout = word_dropout > 0.0
        if self.use_dropout:
            self.dropout = WordDropout(word_dropout)

        self.linear = nn.Linear(2 * self.encoder.config.hidden_size, num_labels)
        self.layer_norm = LayerNorm(2 * self.encoder.config.hidden_size)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        self.multiview = multiview
        self.temperature = temperature

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embed = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]

        if 'emission_mask' in inputs:
            mask = inputs['emission_mask']
            masked_lengths = mask.sum(-1).long()
            masked_reps = torch.zeros_like(embed)
            for i in range(len(mask)):
                masked_reps[i, : masked_lengths[i], :] = (
                    embed[i].masked_select(mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
                )
            reps = masked_reps
        else:
            reps = embed

        if self.use_dropout:
            reps = self.dropout(reps)

        so_head_mask = inputs['so_head_mask']
        batch_size = so_head_mask.shape[0]
        so_emb = reps.masked_select(so_head_mask.unsqueeze(-1)).view(batch_size, -1)

        so_emb = self.layer_norm(so_emb)
        logits = self.linear(so_emb)

        return {'logits': logits}

    def _forward_origin_view(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embed = self.encoder(
            inputs['origin_input_ids'], attention_mask=inputs['origin_attention_mask']
        )[0]

        if 'origin_emission_mask' in inputs:
            mask = inputs['origin_emission_mask']
            masked_lengths = mask.sum(-1).long()
            masked_reps = torch.zeros_like(embed)
            for i in range(len(mask)):
                masked_reps[i, : masked_lengths[i], :] = (
                    embed[i].masked_select(mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
                )
            reps = masked_reps
        else:
            reps = embed

        if self.use_dropout:
            reps = self.dropout(embed)

        so_head_mask = inputs['so_head_mask']
        batch_size = so_head_mask.shape[0]
        so_emb = reps.masked_select(so_head_mask.unsqueeze(-1)).view(batch_size, -1)

        so_emb = self.layer_norm(so_emb)
        logits = self.linear(so_emb)

        return {'logits': logits}

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # noqa
        outputs = self._forward(inputs)

        logits = outputs['logits']
        label_id = inputs['label_id']

        if self.training:
            loss = self._calculate_loss(logits, label_id.view(-1))
            if self.multiview:  # for multiview training
                origin_view_outputs = self._forward_origin_view(inputs)
                origin_view_logits = origin_view_outputs['logits']
                origin_view_loss = self._calculate_loss(origin_view_logits, label_id.view(-1))
                cl_kl_loss = self._calculate_cl_loss(logits, origin_view_logits, T=self.temperature)
                loss = loss + origin_view_loss + cl_kl_loss
            outputs = {'logits': logits, 'loss': loss}
        else:
            predicts = self.decode(logits)
            outputs = {'logits': logits, 'predicts': predicts}

        return outputs

    def _calculate_cl_loss(self, ext_view_logits, origin_view_logits, T=1.0):
        if self.multiview:
            batch_size, num_classes = ext_view_logits.shape
            ext_view_logits = ext_view_logits.detach()
            _loss = (
                F.kl_div(
                    F.log_softmax(origin_view_logits / T, dim=-1),
                    F.softmax(ext_view_logits / T, dim=-1),
                    reduction='none',
                )
                * T
                * T
            )
            loss = _loss.sum() / batch_size
        else:
            loss = 0.0
        return loss

    def _calculate_loss(self, logits, targets):
        loss = self.loss_fn(logits, targets)
        return loss

    def decode(self, logits):  # noqa
        predicts = logits.argmax(-1)
        return predicts
