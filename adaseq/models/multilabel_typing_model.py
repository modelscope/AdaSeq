# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from modelscope.models.builder import MODELS
from torch.nn import BCELoss, BCEWithLogitsLoss

from adaseq.metainfo import Models
from adaseq.models.base import Model
from adaseq.modules.decoders import Decoder, PairwiseCRF
from adaseq.modules.dropouts import WordDropout
from adaseq.modules.embedders import Embedder
from adaseq.modules.encoders import SpanEncoder


class WBCEWithLogitsLoss:
    """Weighed BCE loss, multiply the loss of positive examples with a scaler"""

    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight
        self.loss = BCEWithLogitsLoss(reduction='none')

    def __call__(self, y_pred, y_true):  # noqa: D102
        loss = self.loss(y_pred, y_true.float())

        if self.pos_weight != 1:
            weights_pr = torch.ones_like(y_true).float()  # B x C
            weights_pr[y_true > 0] = self.pos_weight
            loss = (loss * weights_pr).mean()
        else:
            loss = loss.mean()

        return loss


class WBCEWithLogitsLossUFET:
    """Weighed BCE loss, multiply the loss of positive examples with a scaler,
    Apply the trick in Ultra-Fine Entity Typing with Weak Supervision from a Masked Language Model"""

    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight
        self.loss = BCEWithLogitsLoss(reduction='none')

    def __call__(self, y_pred, y_true):  # noqa: D102
        loss = self.loss(y_pred, y_true.float())
        B, L = y_true.shape
        coarse = y_true[:, :9].sum(-1) > 0
        fine = y_true[:, 9:130].sum(-1) > 0
        ultra = y_true[:, 130:].sum(-1) > 0
        a = coarse.view(-1, 1) * torch.ones_like(y_true[:, :9])
        b = fine.view(-1, 1) * torch.ones_like(y_true[:, 9:130])
        c = ultra.view(-1, 1) * torch.ones_like(y_true[:, 130:])
        weight_cfu = torch.cat([a, b, c], -1)
        loss = loss * weight_cfu
        if self.pos_weight != 1:
            weights_pr = torch.ones_like(y_true).float()  # B x C
            weights_pr[y_true > 0] = self.pos_weight
            loss = (loss * weights_pr).mean()
        else:
            loss = loss.mean()

        return loss


@MODELS.register_module(module_name=Models.multilabel_span_typing_model)
class MultiLabelSpanTypingModel(Model):
    """Span based MultiLabel Entity Typing model

    This model is used for multilabel entity typing tasks.

    Args:
        num_labels (int): number of labels
        embedder (Union[Embedder, str], `optional`): embedder used in the model.
            It can be an `Embedder` instance or an embedder config file or an embedder config dict.
        span_encoder_method (str): concat
        word_dropout (float, `optional`): word dropout rate, default `0.0`.
        loss_function (str, `optional'): loss function, default 'BCE',
        class_threshold (float, `optional`): classification threshold default. '0.5'
        use_biaffine (bool, `optional`): whether to use biaffine for span representation, default `False`.
        **kwargs: other arguments
    """

    def __init__(
        self,
        id_to_label: Dict[int, str],
        embedder: Union[Embedder, str],
        span_encoder_method: str = 'concat',
        dropout: float = 0.0,
        word_dropout: bool = False,
        loss_function: str = 'BCE',
        class_threshold: float = 0.5,
        use_biaffine: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(embedder)

        self.span_encoder = SpanEncoder(
            self.embedder.get_output_dim(), span_encoder_method, **kwargs
        )
        self.linear_input_dim = self.span_encoder.output_dim

        self.id_to_label = id_to_label
        self.num_labels = len(id_to_label)
        self.linear = nn.Linear(self.linear_input_dim, self.num_labels)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            if word_dropout:
                self.dropout = WordDropout(dropout)
            else:
                self.dropout = nn.Dropout(dropout)

        self.loss_function_type = loss_function
        self.class_threshold = class_threshold
        self.use_biaffine = use_biaffine

        if self.loss_function_type == 'BCE':
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = BCELoss()

    def forward(  # noqa: D102
        self,
        tokens: Dict[str, Any],
        mention_boundary: torch.LongTensor,
        mention_mask: torch.LongTensor,
        type_ids: Optional[torch.LongTensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # TODO docstring
        # B x M -> B*M x 1
        logits = self._forward(tokens, mention_boundary)
        mask = mention_mask.reshape(-1).unsqueeze(-1)
        if self.training:
            loss = self._calculate_loss(logits, type_ids, mask)
            outputs = {'logits': logits, 'loss': loss}
        else:
            batch_size, max_mention_per_sent = type_ids.shape[0:2]  # TODO: no supervision
            logits = logits.reshape(batch_size, max_mention_per_sent, -1)
            predicts = self.classify(logits, mask)
            outputs = {'logits': logits, 'predicts': predicts}
        return outputs

    def _forward(self, tokens: Dict[str, Any], mention_boundary: torch.LongTensor) -> torch.Tensor:
        # B*M x K
        if self.use_biaffine:
            span_start_reprs, span_end_reprs = self._token2span_encode(tokens, mention_boundary)
            span_reprs = span_start_reprs * span_end_reprs
        else:
            span_reprs = self._token2span_encode(tokens, mention_boundary)

        # B*M x label_num
        logits = self.linear(span_reprs)
        return logits

    def _token2span_encode(
        self, tokens: Dict[str, Any], mention_boundary: torch.LongTensor
    ) -> torch.Tensor:
        embed = self.embedder(**tokens)
        # embed: B x W x K
        if self.use_dropout:
            embed = self.dropout(embed)
        # span_reprs: B x S x D: trans sentence sequence to mentions sequence
        # the shape of boundary_ids is batch_size x 2 x max_mention_num, providing the detected mentions
        span_reprs = self.span_encoder(embed, mention_boundary)
        return span_reprs

    def _calculate_loss(self, logits, targets, mask):
        logits = logits * mask  # B*M x L
        if self.loss_function_type == 'BCE':
            logits = self.sigmoid(logits)  # B*M x L
            targets = targets.reshape(-1, self.num_labels).to(torch.float32)  # B*M x L
            loss = self.loss_fn(logits, targets) * mask
        else:
            raise ValueError('Unsupported loss %s', self.loss_function_type)
        loss = loss.sum() / (mask.sum())
        return loss

    def classify(self, logits, mention_mask):  # noqa: D102
        if self.loss_function_type == 'BCE':
            logits = self.sigmoid(logits)  # B*M x L
            predicts = torch.where(logits > self.class_threshold, 1, 0)  # B*M x L
        else:
            raise ValueError('Unsupported loss %s', self.loss_function_type)

        predicts = predicts.detach().cpu().numpy()
        batch_mention_mask = mention_mask.detach().cpu().numpy()
        batch_mentions = []
        for prediction, mask in zip(predicts, batch_mention_mask):
            mentions = []
            for pred, flag in zip(prediction, mask):
                if flag == 1:
                    types = set(self.id_to_label[i] for i, p in enumerate(pred) if p == 1)
                    mentions.append(types)
            batch_mentions.append(mentions)

        return batch_mentions


@MODELS.register_module(module_name=Models.multilabel_concat_typing_model)
class MultiLabelConcatTypingModel(Model):
    """Concat based Single Mention MultiLabel Entity Typing model

    This model is used for single mention multilabel entity typing tasks.
    Input format [cls] sentence [sep] mention [sep]

    Args:
        labels (List): list of labels
        embedder (Union[Embedder, str], `optional`): embedder used in the model.
            It can be an `Embedder` instance or an embedder config file or an embedder config dict.
        word_dropout (float, `optional`): word dropout rate, default `0.0`.
        loss_function (str, `optional'): loss function, default 'BCE', other options `WBCE`, `WBCE-UFET'
        class_threshold (float, `optional`): classification threshold default. '0.5'
        pos_weight (float, `optional`): multiplier on loss of positive examples, default 1.0.
        decoder (Union[Encoder, str], `optional`), can be linear or pairwise-crf
        **kwargs: other arguments
    """

    def __init__(
        self,
        id_to_label: Dict[int, str],
        embedder: Union[Embedder, Dict] = None,
        dropout: float = 0.0,
        word_dropout: bool = False,
        loss_function: str = 'BCE',
        class_threshold: float = 0.5,
        pos_weight: float = 1.0,
        decoder: Union[Decoder, str] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.id_to_label = id_to_label
        self.num_labels = len(id_to_label)
        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            if isinstance(embedder, dict):
                embedder['drop_special_tokens'] = False
            self.embedder = Embedder.from_config(embedder)

        self.linear_input_dim = self.embedder.get_output_dim()

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            if word_dropout:
                self.dropout = WordDropout(dropout)
            else:
                self.dropout = nn.Dropout(dropout)

        self.loss_function_type = loss_function
        self.class_threshold = class_threshold
        self.pos_weight = pos_weight

        if self.loss_function_type == 'BCE':
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = BCEWithLogitsLoss()
        elif self.loss_function_type == 'WBCE':
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = WBCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif self.loss_function_type == 'WBCE-UFET':
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = WBCEWithLogitsLossUFET(pos_weight=self.pos_weight)

        self.linear = nn.Linear(self.linear_input_dim, self.num_labels)

        self.decoder_type = decoder.pop('type')
        assert self.decoder_type in [
            'pairwise-crf',
            'linear',
        ], 'decoder_type {} unimplemented'.format(self.decoder_type)

        if self.decoder_type == 'pairwise-crf':
            self.decoder = PairwiseCRF(
                labels=[id_to_label[i] for i in range(self.num_labels)], **decoder
            )

    def forward(  # noqa: D102
        self,
        tokens: Dict[str, Any],
        type_ids: Optional[torch.LongTensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # TODO docstring
        # B x M -> B*M x 1
        logits = self._forward(tokens)
        if self.training and type_ids is not None:
            loss = self.loss_fn(logits, type_ids.float())
            outputs = {'logits': logits, 'loss': loss}
        else:
            predicts = self.classify(logits)
            outputs = {'logits': logits, 'predicts': predicts}
        return outputs

    def _forward(self, tokens: Dict[str, Any]) -> torch.Tensor:
        span_reprs = self._token2span_encode(tokens)  # B x K
        # B*M x label_num
        logits = self.linear(span_reprs)
        if self.decoder_type == 'pairwise-crf':
            logits = self.decoder(logits)

        return logits

    def _token2span_encode(self, tokens: Dict[str, Any]) -> torch.Tensor:
        embed = self.embedder(**tokens)
        # embed: B x W x K
        if self.use_dropout:
            embed = self.dropout(embed)
        # use the cls embedding
        return embed[:, 0, :]

    def classify(self, logits):  # noqa: D102
        logits = self.sigmoid(logits)  # B x L
        predicts = []
        for i in range(len(logits)):
            pred_ = (logits[i] > 0.5).nonzero()  # at least predict one
            if len(pred_) < 1:
                pred_ = logits[i].topk(k=1).indices
            types = set(self.id_to_label[i] for i in pred_.view(-1).cpu().numpy())
            predicts.append([types])
        return predicts
