# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models.builder import MODELS
from modelscope.utils.config import ConfigDict

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import Models, Pipelines, Tasks
from adaseq.models.base import Model
from adaseq.modules.decoders import CRF, PartialCRF
from adaseq.modules.dropouts import WordDropout
from adaseq.modules.embedders import Embedder
from adaseq.modules.encoders import Encoder
from adaseq.modules.util import get_tokens_mask


@MODELS.register_module(Tasks.word_segmentation, module_name=Models.sequence_labeling_model)
@MODELS.register_module(Tasks.part_of_speech, module_name=Models.sequence_labeling_model)
@MODELS.register_module(Tasks.named_entity_recognition, module_name=Models.sequence_labeling_model)
class SequenceLabelingModel(Model):
    """Sequence labeling model

    This model is used for sequence labeling tasks.
    Various decoders are supported, including argmax, crf, partial-crf, etc.

    Args:
        num_labels (int): number of labels
        embedder (Union[Embedder, str], `optional`): embedder used in the model.
            It can be an `Embedder` instance or an embedder config file or an embedder config dict.
        dropout (float, `optional`): dropout rate, default `0.0`.
        word_dropout (bool): if `True`, use `WordDropout`.
        use_crf (bool, `optional`): whether to use crf, default `True`.
        **kwargs: other arguments
    """

    pipeline = Pipelines.sequence_labeling_pipeline

    def __init__(
        self,
        id_to_label: Dict[int, str],
        embedder: Union[Embedder, ConfigDict],
        encoder: Optional[Union[Encoder, ConfigDict]] = None,
        dropout: float = 0.0,
        word_dropout: bool = False,
        use_crf: Optional[bool] = True,
        multiview: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        mv_loss_type: Optional[str] = 'kl',
        mv_interpolation: Optional[float] = 0.5,
        partial: Optional[bool] = False,
        chunk: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.id_to_label = id_to_label
        self.num_labels = len(id_to_label)

        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(embedder)
        hidden_size = self.embedder.get_output_dim()

        if encoder is None:
            self.encoder = None
        else:
            if isinstance(encoder, Encoder):
                self.encoder = encoder
            else:
                self.encoder = Encoder.from_config(encoder)
            assert hidden_size == self.encoder.get_input_dim()
            hidden_size = self.encoder.get_output_dim()

        self.linear = nn.Linear(hidden_size, self.num_labels)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            if word_dropout:
                self.dropout = WordDropout(dropout)
            else:
                self.dropout = nn.Dropout(dropout)

        self.use_crf = use_crf
        if use_crf:
            if partial:
                self.crf = PartialCRF(self.num_labels, batch_first=True)
            else:
                self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_LABEL_ID)

        self.multiview = multiview
        self.mv_loss_type = mv_loss_type
        self.temperature = temperature
        self.mv_interpolation = mv_interpolation
        self.chunk = chunk

    def forward(
        self,
        tokens: Dict[str, Any],
        label_ids: Optional[torch.LongTensor] = None,
        meta: Optional[Dict[str, Any]] = None,
        origin_tokens: Optional[Dict[str, Any]] = None,
        origin_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        TODO docstring
        """

        if self.chunk:
            self._flatten(tokens)
            logits = self._forward(tokens)
            logits = logits.view(len(meta), -1, logits.size(-2), logits.size(-1)).max(dim=1)[0]
            logits = logits[:, : label_ids.size(1), :]
            crf_mask = (
                get_tokens_mask(tokens, logits.size(1)) if origin_mask is None else origin_mask
            )

        else:
            logits = self._forward(tokens)
            crf_mask = (
                get_tokens_mask(tokens, logits.size(1)) if origin_mask is None else origin_mask
            )

        if self.training and label_ids is not None:
            loss = self._calculate_loss(logits, label_ids, crf_mask)

            if self.multiview and origin_tokens is not None:  # for multiview training
                origin_view_logits = self._forward(origin_tokens)
                origin_size = origin_view_logits.size(1 if origin_view_logits.dim() == 3 else 0)
                origin_mask_short = crf_mask[..., :origin_size]
                origin_label_ids = label_ids[..., :origin_size]

                origin_view_loss = self._calculate_loss(
                    origin_view_logits, origin_label_ids, origin_mask_short
                )
                logits_short = logits[..., :origin_size, :]
                cl_kl_loss = self._calculate_cl_loss(
                    logits_short, origin_view_logits, origin_mask_short, T=self.temperature
                )
                loss = (
                    self.mv_interpolation * (loss + origin_view_loss)
                    + (1 - self.mv_interpolation) * cl_kl_loss
                )

            outputs = {'logits': logits, 'loss': loss}
        else:
            predicts = self.decode(logits, crf_mask)
            outputs = {'logits': logits, 'predicts': predicts}

        return outputs

    def _forward(self, tokens: Dict[str, Any]) -> torch.Tensor:
        x = self.embedder(**tokens)

        if self.use_dropout:
            x = self.dropout(x)

        if self.encoder is not None:
            mask = get_tokens_mask(tokens, x.size(1))
            x = self.encoder(x, mask)

            if self.use_dropout:
                x = self.dropout(x)

        logits = self.linear(x)
        return logits

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

    def _flatten(
        self,
        tokens: Dict[str, Any],
    ):
        fields = set(tokens.keys())
        for field in fields:
            if len(tokens[field].size()) > 2:
                tokens[field] = tokens[field].flatten(0, 1)
