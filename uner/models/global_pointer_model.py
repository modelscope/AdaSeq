# Auther: xuanjie.wxb
# Date: 2022-07-06 19:07
# Comments:

import sys
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from modelscope.models.builder import MODELS

from uner.metainfo import Models
from uner.models.base import Model
from uner.modules.dropouts import WordDropout
from uner.modules.encoders import Encoder


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
       ref: https://spaces.ac.cn/archives/8265
    """

    def __init__(self,
                 output_dim,
                 merge_mode='add',
                 custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            seq_len = input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack(
            [torch.sin(embeddings),
             torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        embeddings = embeddings.to(inputs.device)
        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings


@MODELS.register_module(module_name=Models.global_pointer_model)
class GlobalPointerModel(Model):

    def __init__(self,
                 num_labels: int,
                 encoder: Union[Encoder, str] = None,
                 token_ffn_out_width=-1,
                 encoder_hidden_size=-1,
                 word_dropout=0.0,
                 **kwargs):
        super(GlobalPointerModel, self).__init__()

        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            self.encoder = Encoder.from_config(
                cfg_dict_or_path=encoder, **kwargs)

        self.token_ffn_out_width = token_ffn_out_width
        self.token_inner_embed_ffn = nn.Linear(encoder_hidden_size,
                                               token_ffn_out_width * 2)
        self.type_score_ffn = nn.Linear(encoder_hidden_size, num_labels * 2)

        self.pos_embed = SinusoidalPositionEmbedding(token_ffn_out_width,
                                                     'zero')
        self.use_dropout = word_dropout > 0.0
        if self.use_dropout:
            self.dropout = WordDropout(word_dropout)

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        embed = self.encoder(
            inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        embed = embed * 10

        if self.use_dropout:
            embed = self.dropout(embed)

        token_inner_embed = self.token_inner_embed_ffn(embed)
        start_token, end_token = token_inner_embed[
            ..., ::2], token_inner_embed[..., 1::2]
        pos = self.pos_embed(token_inner_embed)
        cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)

        def add_position_embedding(in_embed, cos_pos, sin_pos):
            additional_part = torch.stack(
                [-in_embed[..., 1::2], in_embed[..., ::2]], 3)
            additional_part = torch.reshape(additional_part, start_token.shape)
            output_embed = in_embed * cos_pos + additional_part * sin_pos
            return output_embed

        start_token = add_position_embedding(start_token, cos_pos, sin_pos)
        end_token = add_position_embedding(end_token, cos_pos, sin_pos)
        span_score = torch.einsum('bmd,bnd->bmn', start_token,
                                  end_token) / self.token_ffn_out_width**0.5
        typing_score = torch.einsum('bnh->bhn', self.type_score_ffn(embed)) / 2
        entity_score = span_score[:,
                                  None] + typing_score[:, ::2,
                                                       None] + typing_score[:,
                                                                            1::
                                                                            2, :,
                                                                            None]  # [:, None] 增加一个维度
        entity_score = self.add_mask_tril(
            entity_score, mask=inputs['attention_mask'])
        return {'entity_score': entity_score}

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self._forward(inputs)
        if self.training:
            loss = self._calculate_loss(outputs['entity_score'],
                                        inputs['label_matrix'])
            outputs = {'entity_score': outputs['entity_score'], 'loss': loss}
        else:
            predicts = self.decode(outputs['entity_score'])
            outputs = {
                'entity_score': outputs['entity_score'],
                'predicts': predicts
            }
        return outputs

    def _calculate_loss(self, entity_score, targets):
        """
        targets : (batch_size, num_classes, seq_len, seq_len)
        entity_score : (batch_size, num_classes, seq_len, seq_len)
        """
        batch_size, num_classes = entity_score.shape[:2]
        targets = targets.reshape(batch_size * num_classes, -1)
        entity_score = entity_score.reshape(batch_size * num_classes, -1)
        loss = self.multilabel_categorical_crossentropy(targets, entity_score)
        return loss

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            # return x * mask + value * (1 - mask)
            return x * mask + value * ~mask

    def add_mask_tril(self, entity_score, mask):
        entity_score = self.sequence_masking(entity_score, mask, '-inf',
                                             entity_score.ndim - 2)
        entity_score = self.sequence_masking(entity_score, mask, '-inf',
                                             entity_score.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(entity_score), diagonal=-1)
        entity_score = entity_score - mask * 1e12
        return entity_score

    def multilabel_categorical_crossentropy(self, targets, entity_score):
        """
        https://kexue.fm/archives/7359
        """
        entity_score = (1 - 2 * targets
                        ) * entity_score  # -1 -> pos classes, 1 -> neg classes
        entity_score_neg = entity_score - targets * 1e12  # mask the pred outputs of pos classes
        entity_score_pos = (entity_score - (1 - targets) * 1e12
                            )  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(entity_score[..., :1])
        entity_score_neg = torch.cat([entity_score_neg, zeros], dim=-1)
        entity_score_pos = torch.cat([entity_score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(entity_score_neg, dim=-1)
        pos_loss = torch.logsumexp(entity_score_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def decode(self, entity_score):
        score_matrixes = entity_score.detach().cpu().numpy()
        pred_entities_batch = []
        for score_matrix in score_matrixes:
            pred_entities_sent = []
            for t, s, e in zip(*np.where(score_matrix > 0)):
                pred_entities_sent.append((s, e, t))
            pred_entities_batch.append(pred_entities_sent)
        return pred_entities_batch
