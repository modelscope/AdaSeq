# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from modelscope.models.builder import MODELS

from adaseq.metainfo import Models, Pipelines, Tasks
from adaseq.models.base import Model
from adaseq.modules.biaffine import Biaffine
from adaseq.modules.dropouts import WordDropout
from adaseq.modules.embedders import Embedder
from adaseq.modules.encoders import Encoder
from adaseq.modules.util import get_tokens_mask


@MODELS.register_module(Tasks.named_entity_recognition, module_name=Models.biaffine_ner_model)
class BiaffineNerModel(Model):
    """
    Named Entity Recognition as Dependency Parsing (Yu et al., ACL 2020)
    ref: https://aclanthology.org/2020.acl-main.577/
    """

    pipeline = Pipelines.span_based_ner_pipeline

    def __init__(
        self,
        id_to_label: Union[Dict[int, str], List[str]],
        embedder: Optional[Union[Embedder, Dict]] = None,
        encoder: Optional[Union[Encoder, Dict[str, Any]]] = None,
        biaffine_ffnn_size: int = -1,
        biaffine_bias: bool = True,
        multi_label: bool = False,
        flat_ner: bool = True,
        dropout: float = 0.0,
        word_dropout: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.id_to_label = id_to_label
        self.num_labels = len(id_to_label) + 1  # leave 0 as non-entity label
        self.flat_ner = flat_ner
        self.multi_label = multi_label

        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(embedder, **kwargs)
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

        self.scorer = BiaffineScorer(
            hidden_size, biaffine_ffnn_size, self.num_labels * (1 + int(multi_label)), biaffine_bias
        )

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            if word_dropout:
                self.dropout = WordDropout(dropout)
            else:
                self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: Dict[str, Any],
        span_labels: Optional[torch.LongTensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
        tokens (Dict[str, Any]):
            embedder inputs.
        span_labels (torch.LongTensor):
            span labels with size (len, len), span_labels[start][end] == label_id
        meta (Dict[str, Any]):
            meta data of raw inputs.
        """
        x = self.embedder(**tokens)
        mask = get_tokens_mask(tokens, x.size(1))

        if self.use_dropout:
            x = self.dropout(x)

        if self.encoder is not None:
            x = self.encoder(x, mask)
            if self.use_dropout:
                x = self.dropout(x)

        span_scores = self.scorer(x)
        outputs = {'span_scores': span_scores}

        if self.training and span_labels is not None:
            loss = self._calculate_loss(outputs['span_scores'], span_labels, mask)
            outputs.update(loss=loss)
        else:
            predicts = self.decode(outputs['span_scores'], mask)
            outputs.update(predicts=predicts)
        return outputs

    def _calculate_loss(
        self, span_scores: torch.Tensor, span_labels: torch.LongTensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        span_labels : (batch_size, seq_len, seq_len)
        span_scores : (batch_size, seq_len, seq_len, num_classes)
        """
        if self.multi_label:  # TODO why?
            new_scores = span_scores.view(-1, 2)
            new_labels = nn.functional.one_hot(span_labels, self.num_labels)
            mi_mask = torch.triu(mask.unsqueeze(-1).expand_as(span_labels).clone())
            en_mask = mi_mask.unsqueeze(-1).expand_as(new_labels).clone()
            use_labels = new_labels.masked_fill(~en_mask, -100).view(-1)
            loss = nn.functional.cross_entropy(new_scores, use_labels)
        else:
            label_mask = torch.triu(mask.unsqueeze(-1).expand_as(span_labels).clone())
            loss = nn.functional.cross_entropy(
                span_scores.reshape(-1, self.num_labels),
                span_labels.masked_fill(~label_mask, -100).reshape(-1),
            )
        return loss

    def decode(self, span_scores, mask):
        """
        :param span_scores: (b, t, t, c)
        :param mask: (b, t)
        :return:
        """
        batch_size, t_shape = span_scores.shape[0], span_scores.shape[1]
        lengths = mask.sum(-1).tolist()
        # mult = self.multi_label and not self.flat_ner
        if not self.multi_label:
            # (b, t, t)
            type_idxs = span_scores.detach().argmax(dim=-1)
            # (b, t, t)
            span_max_score = (
                span_scores.detach().gather(dim=-1, index=type_idxs.unsqueeze(-1)).squeeze(-1)
            )
        else:
            binary_scores = span_scores.detach().view(-1, 2)  # (b*t*t*c, 2)
            # if not mult:
            #     new_span_scores = binary_scores[..., 1].view(-1, t_shape, t_shape, self.num_labels)
            #     assert new_span_scores.shape[0] == batch_size
            #     type_idxs = new_span_scores.argmax(dim=-1)
            #     span_max_score = (
            #         new_span_scores.gather(dim=-1, index=type_idxs.unsqueeze(-1)).squeeze(1)
            #     )
            # else:
            #     cur = binary_scores.argmax(dim=-1).view(-1, self.num_labels)

            # use the second dim as span_scores
            span_scores = binary_scores[..., 1].view(-1, t_shape, t_shape, self.num_labels)
            assert span_scores.shape[0] == batch_size
            type_idxs = span_scores.argmax(dim=-1)
            span_max_score = span_scores.gather(dim=-1, index=type_idxs.unsqueeze(-1)).squeeze(1)

        final = []
        for span_score, tids, l in zip(span_max_score, type_idxs, lengths):
            cands = []
            for s in range(l):
                for e in range(s, l):
                    type_id = tids[s, e].item()
                    if type_id > 0:
                        # we do not have non-entity label, we set index 0 as non-entity label
                        # so we substract 1 on type_id
                        cands.append((s, e, type_id - 1, span_score[s, e].item()))

            pre_res = []
            for s, e, cls, _ in sorted(cands, key=lambda x: x[3], reverse=True):
                for pred_s, pred_e, _ in pre_res:
                    # contraint on the token level, so convert sub-token indices to token
                    if s < pred_s <= e < pred_e or pred_s < s <= pred_e < e:
                        break  # for both nested and flat ner no clash is allowed
                    if self.flat_ner and (s <= pred_s <= pred_e <= e or pred_s <= s <= e <= pred_e):
                        break  # for flat ner nested mentions are not allowed
                else:
                    pre_res.append((s, e + 1, self.id_to_label[cls]))
            final.append(pre_res)

        return final


class BiaffineScorer(nn.Module):
    """Biaffine scorer."""

    def __init__(
        self,
        input_size: int,
        ffnn_size: int,
        num_cls: int,
        # ffnn_drop: float = 0.33,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._act = nn.ELU()
        self.mlp_start = nn.Linear(input_size, ffnn_size)
        self.mlp_end = nn.Linear(input_size, ffnn_size)
        self.span_biaff = Biaffine(ffnn_size, num_cls, bias=(bias, bias))
        # self._dropout = WordDropout(ffnn_drop)

    def forward(self, enc_hn: torch.Tensor) -> torch.Tensor:
        """biaffine attention scores."""
        start_feat = self._act(self.mlp_start(enc_hn))
        end_feat = self._act(self.mlp_end(enc_hn))

        # if self.training:
        #     start_feat = self._dropout(start_feat)
        #     end_feat = self._dropout(end_feat)

        # (bz, len, len, num_lbl)
        span_score = self.span_biaff(start_feat, end_feat)
        return span_score
