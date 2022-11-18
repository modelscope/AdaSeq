# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models.builder import MODELS
from modelscope.utils.config import ConfigDict

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import Models
from adaseq.models.base import Model
from adaseq.modules.decoders import CRF, PartialCRF
from adaseq.modules.dropouts import WordDropout
from adaseq.modules.encoders import Encoder


@MODELS.register_module(module_name=Models.sequence_labeling_model)
class SequenceLabelingModel(Model):
    """Sequence labeling model

    This model is used for sequence labeling tasks.
    Various decoders are supported, including argmax, crf, partial-crf, etc.

    Args:
        num_labels (int): number of labels
        encoder (Union[Encoder, str], `optional`): encoder used in the model.
            It can be an `Encoder` instance or an encoder config file or an encoder config dict.
        word_dropout (float, `optional`): word dropout rate, default `0.0`.
        use_crf (bool, `optional`): whether to use crf, default `True`.
        **kwargs: other arguments
    """

    def __init__(
        self,
        num_labels: int,
        encoder: Union[Encoder, str, ConfigDict] = None,
        word_dropout: Optional[float] = 0.0,
        use_crf: Optional[bool] = True,
        multiview: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        mv_loss_type: Optional[str] = 'kl',
        mv_interpolation: Optional[float] = 0.5,
        partial: Optional[bool] = False,
        **kwargs
    ):
        super(SequenceLabelingModel, self).__init__()
        self.num_labels = num_labels
        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            self.encoder = Encoder.from_config(cfg_dict_or_path=encoder, **kwargs)
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_labels)

        self.use_dropout = word_dropout > 0.0
        if self.use_dropout:
            self.dropout = WordDropout(word_dropout)

        self.use_crf = use_crf
        if use_crf:
            if partial:
                self.crf = PartialCRF(num_labels, batch_first=True)
            else:
                self.crf = CRF(num_labels, batch_first=True)
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_LABEL_ID)

        self.multiview = multiview
        self.mv_loss_type = mv_loss_type
        self.temperature = temperature
        self.mv_interpolation = mv_interpolation

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embed = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]

        if self.use_dropout:
            embed = self.dropout(embed)

        logits = self.linear(embed)

        if 'emission_mask' in inputs:
            mask = inputs['emission_mask']
            masked_lengths = mask.sum(-1).long()
            masked_logits = torch.zeros_like(logits)
            for i in range(len(mask)):
                masked_logits[i, : masked_lengths[i], :] = (
                    logits[i].masked_select(mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
                )
            logits = masked_logits

        return {'logits': logits}

    def _forward_origin_view(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embed = self.encoder(
            inputs['origin_input_ids'], attention_mask=inputs['origin_attention_mask']
        )[0]

        if self.use_dropout:
            embed = self.dropout(embed)

        logits = self.linear(embed)

        if 'origin_emission_mask' in inputs:
            mask = inputs['origin_emission_mask']
            masked_lengths = mask.sum(-1).long()
            masked_logits = torch.zeros_like(logits)
            for i in range(len(mask)):
                masked_logits[i, : masked_lengths[i], :] = (
                    logits[i].masked_select(mask[i].unsqueeze(-1)).view(masked_lengths[i], -1)
                )
            logits = masked_logits

        return {'logits': logits}

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # noqa
        outputs = self._forward(inputs)

        logits = outputs['logits']
        label_ids = inputs['label_ids']
        seq_lens = inputs['emission_mask'].sum(-1).long()
        mask = (
            torch.arange(inputs['emission_mask'].shape[1], device=seq_lens.device)[None, :]
            < seq_lens[:, None]
        )

        if self.training:
            loss = self._calculate_loss(logits, label_ids, mask)
            if self.multiview:  # for multiview training
                origin_view_outputs = self._forward_origin_view(inputs)
                origin_view_logits = origin_view_outputs['logits']
                origin_view_loss = self._calculate_loss(origin_view_logits, label_ids, mask)
                if self.mv_loss_type == 'kl':
                    cl_kl_loss = self._calculate_cl_loss(
                        logits, origin_view_logits, mask, T=self.temperature
                    )
                    loss = (
                        self.mv_interpolation * (loss + origin_view_loss)
                        + (1 - self.mv_interpolation) * cl_kl_loss
                    )
                elif self.mv_loss_type == 'crf_kl':
                    cl_kl_loss = self._calculate_cl_loss(
                        logits, origin_view_logits, mask, T=self.temperature
                    )
                    loss = (
                        self.mv_interpolation * (loss + origin_view_loss)
                        + (1 - self.mv_interpolation) * cl_kl_loss
                    )
            outputs = {'logits': logits, 'loss': loss}
        else:
            predicts = self.decode(logits, mask)
            outputs = {'logits': logits, 'predicts': predicts}

        return outputs

    def _calculate_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.use_crf:
            targets = targets * mask
            loss = -self.crf(logits, targets, reduction='mean', mask=mask)
        else:
            loss = self.loss_fn(logits.transpose(1, 2), targets)
        return loss

    def _calculate_cl_loss(self, ext_view_logits, origin_view_logits, mask, T=1.0):
        if self.multiview:
            batch_size, max_seq_len, num_classes = ext_view_logits.shape
            ext_view_logits = ext_view_logits.detach()
            if self.mv_loss_type == 'kl':
                _loss = (
                    F.kl_div(
                        F.log_softmax(origin_view_logits / T, dim=-1),
                        F.softmax(ext_view_logits / T, dim=-1),
                        reduction='none',
                    )
                    * mask.unsqueeze(-1)
                    * T
                    * T
                )
            elif self.mv_loss_type == 'crf_kl':
                if self.use_crf:
                    origin_view_log_posterior = self.crf.compute_posterior(origin_view_logits, mask)
                    ext_view_log_posterior = self.crf.compute_posterior(ext_view_logits, mask)
                    _loss = (
                        F.kl_div(
                            F.log_softmax(origin_view_log_posterior / T, dim=-1),
                            F.softmax(ext_view_log_posterior / T, dim=-1),
                            reduction='none',
                        )
                        * mask.unsqueeze(-1)
                        * T
                        * T
                    )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            loss = _loss.sum() / batch_size
        else:
            loss = 0.0
        return loss

    def decode(  # noqa: D102
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> Union[List, torch.LongTensor]:
        if self.use_crf:
            predicts = self.crf.decode(logits, mask=mask).squeeze(0)
        else:
            predicts = logits.argmax(-1)
        return predicts
