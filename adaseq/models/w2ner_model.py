# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from modelscope.models.builder import MODELS

from adaseq.metainfo import Models, Tasks
from adaseq.models.base import Model
from adaseq.modules.biaffine import Biaffine
from adaseq.modules.embedders import Embedder
from adaseq.modules.encoders import CnnEncoder, LstmEncoder
from adaseq.modules.util import get_tokens_mask


class LayerNorm(nn.Module):  # noqa: D101
    def __init__(
        self,
        input_dim,
        cond_dim=0,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='xaiver',
        **kwargs,
    ):
        super().__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(
                    in_features=self.cond_dim, out_features=self.hidden_units, bias=False
                )
            if self.center:
                self.beta_dense = nn.Linear(
                    in_features=self.cond_dim, out_features=input_dim, bias=False
                )
            if self.scale:
                self.gamma_dense = nn.Linear(
                    in_features=self.cond_dim, out_features=input_dim, bias=False
                )

        self.initialize_weights()

    def initialize_weights(self):  # noqa: D102
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):  # noqa: D102
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class MLP(nn.Module):  # noqa: D101
    def __init__(self, n_in, n_out, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # noqa: D102
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):  # noqa: D101
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0.0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(biaffine_size, cls_num, bias=(True, True))
        if channels > 0:
            self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
            self.linear = nn.Linear(ffnn_hid_size, cls_num)
        else:
            self.mlp_rel = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z=None):  # noqa: D102
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)
        if z is not None and self.mlp_rel is not None:
            z = self.dropout(self.mlp_rel(z))
            o2 = self.linear(z)
            o1 += o2
        return o1


@MODELS.register_module(Tasks.named_entity_recognition, module_name=Models.w2ner_model)
class W2NerModel(Model):
    """W2 NER model.
    ref: https://arxiv.org/abs/2112.10070
    ref: https://github.com/ljynlp/W2NER
    """

    def __init__(
        self,
        id_to_label: Dict[int, str],
        embedder: Union[Embedder, Dict[str, Any]],
        lstm_encoder: Union[LstmEncoder, Dict[str, Any]] = None,
        cnn_encoder: Optional[Union[CnnEncoder, Dict[str, Any]]] = None,
        # use_bert_last_4_layers: Optional[bool] = False,
        dist_emb_size: int = 20,
        type_emb_size: int = 20,
        biaffine_size: int = 768,
        ffnn_hid_size: int = 384,
        emb_dropout: float = 0.5,
        out_dropout: float = 0.33,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.id_to_label = {int(k): v for k, v in id_to_label.items()}
        num_labels = len(id_to_label)
        self.num_classes = num_labels + 2

        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(cfg_dict_or_path=embedder)
        hidden_size = self.embedder.get_output_dim()

        if isinstance(lstm_encoder, LstmEncoder):
            self.lstm_encoder = lstm_encoder
        elif isinstance(lstm_encoder, dict):
            if 'input_size' not in lstm_encoder:
                lstm_encoder['input_size'] = hidden_size
            self.lstm_encoder = LstmEncoder(**lstm_encoder)
            assert hidden_size == self.lstm_encoder.get_input_dim()
            hidden_size = self.lstm_encoder.get_output_dim()
        else:
            self.lstm_encoder = None

        self.cln = LayerNorm(hidden_size, hidden_size, conditional=True)

        self.dis_embs = nn.Embedding(20, dist_emb_size)
        self.reg_embs = nn.Embedding(3, type_emb_size)

        conv_input_size = hidden_size + dist_emb_size + type_emb_size
        if isinstance(cnn_encoder, CnnEncoder):
            self.convLayer = cnn_encoder
        elif isinstance(cnn_encoder, dict):
            default_hp = dict(
                input_size=conv_input_size,
                channels=128,
                dilation=[1, 2, 3],
                dropout=0.5,
            )
            for param in default_hp:
                if param not in cnn_encoder:
                    cnn_encoder[param] = default_hp[param]
            self.convLayer = CnnEncoder(**cnn_encoder)
        else:
            self.convLayer = None  # for model testcase

        self.predictor = CoPredictor(
            self.num_classes,
            hidden_size,
            biaffine_size,
            0 if self.convLayer is None else self.convLayer.get_output_dim(),
            ffnn_hid_size,
            out_dropout,
        )

        self.use_dropout = emb_dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(emb_dropout)

        self.criterion = nn.CrossEntropyLoss()

    def forward(  # noqa: D102
        self,
        tokens: Dict[str, Any],
        grid_mask2d: torch.LongTensor,
        dist_inputs: torch.LongTensor,
        sent_length: torch.LongTensor,
        grid_labels: Optional[torch.LongTensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        x = self.embedder(**tokens)
        mask = get_tokens_mask(tokens, x.size(1))

        if self.use_dropout:
            x = self.dropout(x)

        if self.lstm_encoder is not None:
            x = self.lstm_encoder(x, mask)

        cln = self.cln(x.unsqueeze(2), x)  # 这行会影响随机性，原因未知

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        if self.convLayer is not None:
            conv_outputs = self.convLayer(conv_inputs)
            conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        else:
            conv_outputs = None

        scores = self.predictor(x, x, conv_outputs)
        outputs = {'entity_score': scores}

        if self.training:
            loss = self.criterion(scores[grid_mask2d.clone()], grid_labels[grid_mask2d.clone()])
            outputs['loss'] = loss
        else:
            predictions = torch.argmax(scores.detach(), -1)
            predicts = self.decode(predictions.cpu().numpy(), sent_length.cpu().numpy())
            outputs['predicts'] = predicts

        return outputs

    def decode(self, outputs, length):  # noqa: D102
        class Node:
            def __init__(self):
                self.THW = []  # [(tail, type)]
                self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

        decode_entities = []
        q = deque()
        for instance, l in zip(outputs, length):
            predicts = []
            nodes = [Node() for _ in range(l)]
            for cur in reversed(range(l)):
                heads = []
                for pre in range(cur + 1):
                    # THW
                    if instance[cur, pre] > 1:
                        nodes[pre].THW.append((cur, instance[cur, pre]))
                        heads.append(pre)
                    # NNW
                    if pre < cur and instance[pre, cur] == 1:
                        # cur node
                        for head in heads:
                            nodes[pre].NNW[(head, cur)].add(cur)
                        # post nodes
                        for head, tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                nodes[pre].NNW[(head, tail)].add(cur)
                # entity
                for tail, type_id in nodes[cur].THW:
                    if cur == tail:
                        predicts.append(([cur], type_id))
                        continue
                    q.clear()
                    q.append([cur])
                    while len(q) > 0:
                        chains = q.pop()
                        for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                            if idx == tail:
                                predicts.append((chains + [idx], type_id))
                            else:
                                q.append(chains + [idx])
            cur_tuples = set()
            for x in predicts:
                cur_indexes, cur_type_id = x
                cur_type = self.id_to_label[cur_type_id - 2]
                start, end = cur_indexes[0], cur_indexes[-1]
                cur_tuples.add((start, end + 1, cur_type))
            decode_entities.append(list(cur_tuples))

        return decode_entities
