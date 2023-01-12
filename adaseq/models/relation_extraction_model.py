# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models.builder import MODELS
from torch.nn import LayerNorm

from adaseq.metainfo import Models, Tasks
from adaseq.models.base import Model
from adaseq.modules.dropouts import WordDropout
from adaseq.modules.embedders import Embedder


@MODELS.register_module(Tasks.relation_extraction, module_name=Models.relation_extraction_model)
class RelationExtractionModel(Model):
    """Relation extraction model

    This model is used for relation extraction tasks.
    """

    def __init__(
        self,
        id_to_label: Dict[int, str],
        embedder: Union[Embedder, str],
        dropout: float = 0.0,
        word_dropout: bool = False,
        multiview: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        **kwargs
    ):
        super(RelationExtractionModel, self).__init__(**kwargs)
        self.id_to_label = id_to_label
        self.num_labels = len(id_to_label)
        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(embedder)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            if word_dropout:
                self.dropout = WordDropout(dropout)
            else:
                self.dropout = nn.Dropout(dropout)

        hidden_size = self.embedder.get_output_dim()
        self.linear = nn.Linear(2 * hidden_size, self.num_labels)
        self.layer_norm = LayerNorm(2 * hidden_size)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        self.multiview = multiview
        self.temperature = temperature

    def _forward(self, tokens: Dict[str, Any], so_head_mask: torch.Tensor) -> torch.Tensor:
        embed = self.embedder(**tokens)

        if self.use_dropout:
            embed = self.dropout(embed)

        batch_size = so_head_mask.shape[0]
        so_emb = embed.masked_select(so_head_mask.unsqueeze(-1)).view(batch_size, -1)

        so_emb = self.layer_norm(so_emb)
        logits = self.linear(so_emb)

        return logits

    def forward(  # noqa
        self,
        tokens: Dict[str, Any],
        so_head_mask: torch.BoolTensor,
        meta: Dict[str, Any],
        label_id: Optional[torch.LongTensor] = None,
        origin_tokens: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # TODO: add docstring
        logits = self._forward(tokens, so_head_mask)

        if self.training:
            loss = self._calculate_loss(logits, label_id)
            if self.multiview and origin_tokens is not None:  # for multiview training
                origin_logits = self._forward(origin_tokens, so_head_mask)
                origin_loss = self._calculate_loss(origin_logits, label_id)
                cl_kl_loss = self._calculate_cl_loss(logits, origin_logits, T=self.temperature)
                loss = loss + origin_loss + cl_kl_loss
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
