# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from adaseq.metainfo import Decoders

from .base import DECODERS, Decoder


def make_glove_embed(tokens, glove_path, embed_dim=100):
    """Utils function to obtain glove embedding"""

    glove = {}
    vecs = []  # use to produce unk

    i = 0
    # load glove
    with open(os.path.join(glove_path), 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            i += 1

            split_line = line.split()
            word = split_line[0]
            embed_str = split_line[1:]
            try:
                # ignore error
                embed_float = [float(i) for i in embed_str]
            except Exception:
                continue

            if word not in glove:
                glove[word] = embed_float
                vecs.append(embed_float)

    unk = np.zeros(embed_dim)
    print('error lines: {}'.format(i - len(vecs)))

    print('loading glove to task vocab')
    # load glove to task vocab
    embed = []
    for i in tqdm(tokens):
        word = i.lower().split('_')
        sub_emb = []
        for sub in word:
            if sub in glove:
                sub_emb.append(glove[sub])
            else:
                sub_emb.append(unk)
        emb = np.mean(sub_emb, 0).tolist()
        embed.append(emb)

    final_embed = np.array(embed, dtype=np.float)
    return final_embed


def make_tencent_embed(tokens, tencent_path, embed_dim=200):
    """Utils function to obtain tencent zh embedding"""
    glove = {}
    import jieba

    i = 0
    # load glove
    with open(os.path.join(tencent_path), 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            i += 1
            if i == 1:
                continue

            split_line = line.split()
            word = split_line[0]
            embed_str = split_line[1:]
            try:
                # ignore error
                embed_float = [float(i) for i in embed_str]
            except Exception:
                continue

            if word not in glove:
                glove[word] = embed_float

    unk = np.zeros(embed_dim)

    print('loading tencent to task vocab')

    embed = []
    for i in tqdm(tokens):
        word = i.replace('(', '')
        word = word.replace(')', '')

        sub_emb = []
        if word in glove:
            sub_emb.append(glove[word])
        else:
            seg_list = jieba.cut(word, cut_all=True)
            for ww in seg_list:
                if ww in glove:
                    sub_emb.append(glove[ww])
                else:
                    sub_emb.append(unk)

        emb = np.mean(sub_emb, 0).tolist()

        embed.append(emb)

    final_embed = np.array(embed, dtype=np.float)

    return final_embed


def get_label_emb(
    labels,
    label_emb_type='glove',
    label_emb_dim=300,
    source_emb_file_path=None,
    target_emb_dir=None,
    target_emb_name='label.emb',
):
    """produce label embeddings"""
    if target_emb_dir is None:
        print('target emb is not given, returning random embedding')
        torch.randn(len(labels), label_emb_dim)

    os.makedirs(target_emb_dir, exist_ok=True)
    emb_save_path = os.path.join('{}/{}'.format(target_emb_dir, target_emb_name))
    if label_emb_type == 'tencent':
        if os.path.exists(emb_save_path):
            label_emb = torch.from_numpy(pickle.load(open(emb_save_path, 'rb'))).float()
        else:
            print(
                'no pre-processed emb file found, using source emb file to produce label embeddings'
            )
            assert source_emb_file_path is not None, 'require source emb file (tencent)'
            label_emb = make_tencent_embed(
                labels, tencent_path=source_emb_file_path, embed_dim=label_emb_dim
            )
            pickle.dump(label_emb, open(os.path.join(emb_save_path), 'wb'))
            label_emb = torch.from_numpy(label_emb).float()

    elif label_emb_type == 'glove':
        if os.path.exists(emb_save_path):
            label_emb = torch.from_numpy(pickle.load(open(emb_save_path, 'rb'))).float()
        else:
            print(
                'no pre-processed emb file found, using source emb file to produce label embeddings'
            )
            assert source_emb_file_path is not None, 'require source emb file (glove)'
            label_emb = make_glove_embed(
                labels, glove_path=source_emb_file_path, embed_dim=label_emb_dim
            )
            pickle.dump(label_emb, open(os.path.join(emb_save_path), 'wb'))
            label_emb = torch.from_numpy(label_emb).float()

    else:
        raise NotImplementedError

    return label_emb


class SimpleFeedForwardLayer(nn.Module):
    """2-layer feed forward"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        activation: Optional[nn.Module] = None,
        dropout_rate: float = 0,
    ):

        super(SimpleFeedForwardLayer, self).__init__()
        self.linear_projection1 = nn.Linear(input_dim, (input_dim + output_dim) // 2, bias=bias)
        self.linear_projection_mid1 = nn.Linear(
            (input_dim + output_dim) // 2, (input_dim + output_dim) // 2, bias=bias
        )
        self.linear_projection_mid2 = nn.Linear(
            (input_dim + output_dim) // 2, (input_dim + output_dim) // 2, bias=bias
        )

        self.linear_projection2 = nn.Linear((input_dim + output_dim) // 2, output_dim, bias=bias)
        self.activation = activation if activation else nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D102

        inputs = self.activation(self.dropout(self.linear_projection1(inputs)))
        inputs = self.linear_projection2(inputs)
        return inputs


class LabelTransformLayerMLP(nn.Module):
    """MLP on label embeddings to parameterize potentials"""

    def __init__(self, emb_dim, dropout_rate):
        super(LabelTransformLayerMLP, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        act = nn.Tanh()

        self.label_emb_view = nn.ModuleList(
            [
                SimpleFeedForwardLayer(
                    input_dim=self.emb_dim,
                    output_dim=self.emb_dim,
                    bias=False,
                    activation=act,
                    dropout_rate=self.dropout_rate,
                )
                for i in range(4)
            ]
        )

    def forward(self, label_emb):  # noqa: D102

        return (
            self.label_emb_view[0](label_emb).unsqueeze(2),
            self.label_emb_view[1](label_emb).unsqueeze(2),
            self.label_emb_view[2](label_emb).unsqueeze(2),
            self.label_emb_view[3](label_emb).unsqueeze(2),
        )


@DECODERS.register_module(module_name=Decoders.pairwise_crf)
class PairwiseCRF(Decoder):
    """
    The module proposed in the EMNLP2022 Paper to model label correlations,
     require label embeddings and use mean-field variational inference.
    "Modeling Label Correlations for Ultra-Fine Entity Typing with Neural Pairwise Conditional Random Field"
    Inference also based on the paper "Regularized Frank-Wolfe for Dense CRFs"
    Args:
        labels: list of labels, for building the label embeddings.
        label_emb_dim: number of label embedding dimention.
        label_emb_type: glove (en) / tencent (zh).
        source_emb_file_path: path of label embed file, e.g. glove.6B.300.txt.
        target_emb_dir: dir to save label embedding.
        target_emb_name: name of saved label embedding.
        pairwise_factor: important args to balancing unary and pairwise potentials.
        two_potential: whether to use two pairwise potential.
        sign_trick: whether to use two pairwise potential.
        mfvi_step_size: step size in Regularized Frank-Wolfe, default 1.0.
        mfvi_scaler: temperature, default 1.0.
        mfvi_iteration: number of iterations for mean-field variational inference.
        return_logits: true if reture logits, otherwise return probabilities.
        label_dropout: label embedding dropout rate.
        **kwargs:
    """

    def __init__(
        self,
        labels,
        label_emb_dim: int = 300,
        label_emb_type: str = 'glove',
        source_emb_file_path: Optional[str] = None,
        target_emb_dir: Optional[str] = None,
        target_emb_name: str = 'label.emb',
        pairwise_factor: int = 20,
        two_potential: bool = True,
        sign_trick: bool = True,
        mfvi_step_size: float = 1.0,
        mfvi_scaler: float = 1.0,
        mfvi_iteration: int = 3,
        return_logits: bool = True,
        label_dropout: float = 0.0,
        **kwargs
    ):
        super(PairwiseCRF, self).__init__()

        label_emb = get_label_emb(
            labels,
            label_emb_type=label_emb_type,
            label_emb_dim=label_emb_dim,
            source_emb_file_path=source_emb_file_path,
            target_emb_dir=target_emb_dir,
            target_emb_name=target_emb_name,
        )

        self.n_label, self.emb_dim = label_emb.size()
        assert self.emb_dim == label_emb_dim
        self.two_potential = two_potential
        self.pairwise_factor = pairwise_factor
        self.mfvi_step_size = mfvi_step_size
        self.mfvi_scaler = mfvi_scaler
        self.mfvi_iteration = mfvi_iteration
        self.sign_trick = sign_trick
        self.return_logits = return_logits
        self.label_dropout = label_dropout

        self.label_emb = nn.Parameter(label_emb / self.pairwise_factor, requires_grad=True)

        self.domain = 2
        self.sigmoid = nn.Sigmoid()
        self.mfvi_iteration = mfvi_iteration
        self.is_cuda = torch.cuda.is_available()
        self.label_emb_transform = LabelTransformLayerMLP(self.emb_dim, self.label_dropout)
        self.label_view = None

    def forward(self, logits):
        """Mean-field variational Inference"""

        # B = 3

        B, L = logits.size()

        logits_zero = (
            torch.zeros(B, self.n_label).cuda() if self.is_cuda else torch.zeros(B, self.n_label)
        )  # B x L
        q_logits = torch.cat([logits_zero.unsqueeze(0), logits.unsqueeze(0)], 0)  # 2 x B x L
        q_logits = q_logits / self.mfvi_scaler
        l_1 = q_logits[1]
        q_dist = q_logits.softmax(0)  # 2 x B x L

        # 1st Term, Unary Energy Ψu(yi):
        #     measures the cost if the label assignment disagrees with the initial classifier.
        # obtain unary logits
        # obtain unary potential, the negative log likelihood of sigmoid logits
        E1, E2, E3, E4 = self.label_emb_transform(self.label_emb)  # L x D x 4

        if self.two_potential:
            label_view1 = torch.cat([E1, E1, E2, E2], 2)
            label_view2 = torch.cat([E1, E2, E1, E2], 2)
        else:
            label_view1 = torch.cat([E1, E2, E3, E4], 2)
            label_view2 = torch.cat([E1, E3, E2, E4], 2)

        for _ in range(self.mfvi_iteration):

            # 2nd Term, Pairwise Energy Ψp(yi, yj):
            #     measures the cost if two similar label (e.g. label semantics) take different labels.
            q_dist_j_0 = q_dist[0]
            q_dist_j_1 = q_dist[1]
            q_logits_i_0 = q_logits[0]
            q_logits_i_1 = q_logits[1]

            dists_mat = torch.cat(
                [
                    q_dist_j_0.unsqueeze(0),
                    q_dist_j_1.unsqueeze(0),
                    q_dist_j_0.unsqueeze(0),
                    q_dist_j_1.unsqueeze(0),
                ],
                0,
            )  # 4 x B x l
            tmp = torch.einsum('vbl,ldv->vbd', dists_mat, label_view1)
            update_logits = torch.einsum('vbd,ldv->vbl', tmp, label_view2)

            if self.sign_trick:
                l_0 = q_logits_i_0 + update_logits[0] - update_logits[1]
                l_1 = q_logits_i_1 - update_logits[2] + update_logits[3]
            else:
                l_0 = q_logits_i_0 + update_logits[0] + update_logits[1]
                l_1 = q_logits_i_1 + update_logits[2] + update_logits[3]

            new_q_dist = (
                torch.cat([l_0.unsqueeze(0), l_1.unsqueeze(0)], dim=0) / self.mfvi_scaler
            ).softmax(0)
            q_dist = (1 - self.mfvi_step_size) * q_dist + self.mfvi_step_size * new_q_dist

        if self.return_logits:
            return l_1 / self.mfvi_scaler
        else:
            return q_dist[1]  # positive probs B x L
