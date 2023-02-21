# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from modelscope.models.builder import MODELS

from adaseq.data.constant import PAD_LABEL_ID
from adaseq.metainfo import Models, Pipelines, Tasks
from adaseq.models.base import Model
from adaseq.modules.decoders.crf import CRFwithConstraints
from adaseq.modules.embedders import Embedder
from adaseq.modules.encoders import SpanEncoder
from adaseq.modules.util import get_tokens_mask


@MODELS.register_module(Tasks.entity_typing, module_name=Models.twostage_ner_model)
@MODELS.register_module(Tasks.named_entity_recognition, module_name=Models.twostage_ner_model)
class TwoStageNERModel(Model):
    """Two stage ner model

    This model is used for two-stage few-shot ner model with pre-trained NER model (see pretraining_model module).

    Args:
        typing_id_to_label (dict): id2label for typing
        ident_id_to_label (dict): id2label for mention identification
        embedder (Union[Encoder, str], `optional`): embedder used in the model.
            It can be an `embedder` instance or an embedder config file or an encoder config dict.
        span_encoder_method (str): concat
        word_dropout (float, `optional`): word dropout rate, default `0.0`.
        use_biaffine (bool, `optional`): whether to use biaffine for span representation, default `False`.
        **kwargs: other arguments
    """

    pipeline = Pipelines.span_based_ner_pipeline

    def __init__(
        self,
        typing_id_to_label: Dict[int, str],
        ident_id_to_label: Dict[int, str],
        embedder: Union[Embedder, str],
        use_crf: bool = True,
        load_parts: str = None,
        model_helper_path: str = None,
        span_encoder_method: str = 'concat',
        word_dropout: Optional[float] = 0.0,
        use_biaffine: bool = False,
        **kwargs
    ):
        super(TwoStageNERModel, self).__init__(**kwargs)
        self.ident_num_labels = len(ident_id_to_label)
        self.typing_num_labels = len(typing_id_to_label)
        self.ident_id_to_label = {int(k): v for k, v in ident_id_to_label.items()}
        self.typing_id_to_label = {int(k): v for k, v in typing_id_to_label.items()}
        self.use_biaffine = use_biaffine

        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(embedder)

        self.span_encoder = SpanEncoder(
            self.embedder.config.hidden_size,
            span_encoder_method,
            use_biaffine=self.use_biaffine,
            **kwargs
        )

        self.ident_linear = nn.Linear(self.embedder.get_output_dim(), self.ident_num_labels)
        self.typing_linear = nn.Linear(self.span_encoder.output_dim, self.typing_num_labels)

        self.split_space_ident = None
        self.split_space_typing = None

        if load_parts is not None and model_helper_path is not None:
            self.load_from_pretrained(load_parts, model_helper_path)

        self.use_dropout = word_dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(word_dropout)

        self.use_crf = use_crf

        if self.use_crf:
            ident_id2label_list = [v for k, v in ident_id_to_label.items()]
            self.crf = CRFwithConstraints(
                ident_id2label_list, batch_first=True, add_constraint=True
            )
        else:
            self.ident_loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_LABEL_ID)

        self.typing_loss_fn = nn.CrossEntropyLoss(reduction='none')

    def load_from_pretrained(self, loaded_parts, model_helper_path):
        """load pretrained model"""

        loaded_list = loaded_parts.split('I')

        pretrained_state_dict = torch.load(model_helper_path)

        if 'e' in loaded_list:
            self.embedder.load_state_dict(pretrained_state_dict['embedder'])

        if 'i' in loaded_list:
            self.ident_linear.load_state_dict(pretrained_state_dict['ident_linear'])

        if 'si' in loaded_list:
            self.split_space_ident = nn.Linear(
                self.embedder.config.hidden_size, self.embedder.config.hidden_size
            )
            self.split_space_ident.load_state_dict(pretrained_state_dict['split_space_ident'])

        if 'st' in loaded_list:
            self.split_space_typing = nn.Linear(
                self.embedder.config.hidden_size, self.embedder.config.hidden_size
            )
            self.split_space_typing.load_state_dict(pretrained_state_dict['split_space_typing'])

    def _forward_embedder(self, tokens: Dict[str, Any]) -> torch.Tensor:
        token_embed = self.embedder(**tokens)

        if self.use_dropout:
            token_embed = self.dropout(token_embed)

        return token_embed

    def _forward_ident(self, token_embed: torch.Tensor) -> torch.Tensor:

        # for entity detection
        ident_embed = token_embed
        if self.split_space_ident is not None:
            ident_embed = self.split_space_ident(token_embed)

        ident_logits = self.ident_linear(ident_embed)

        return ident_logits

    def _forward_typing(
        self, token_embed: torch.Tensor, mention_boundary: torch.Tensor
    ) -> torch.Tensor:

        # for entity typing
        typing_embed = token_embed
        if self.split_space_typing is not None:
            typing_embed = self.split_space_typing(token_embed)

        # B*M x K
        span_reprs = self.span_encoder(typing_embed, mention_boundary)
        typing_logits = self.typing_linear(span_reprs)

        return typing_logits

    def forward(  # noqa: D102
        self,
        tokens: Dict[str, Any],
        mention_boundary: Optional[torch.LongTensor] = None,
        mention_mask: Optional[torch.LongTensor] = None,
        ident_ids: Optional[torch.LongTensor] = None,
        type_ids: Optional[torch.LongTensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # TODO docstring

        token_embed = self._forward_embedder(tokens)
        ident_mask = get_tokens_mask(tokens, token_embed.size(1))

        if self.training:
            ident_logits = self._forward_ident(token_embed)
            typing_logits = self._forward_typing(token_embed, mention_boundary)
            logits = (ident_logits, typing_logits)
            loss = self._calculate_loss(logits, ident_ids, type_ids, ident_mask, mention_mask)
            outputs = {'logits': logits, 'loss': loss}
        else:
            predicts = self.decode(token_embed, ident_mask)
            outputs = {'predicts': predicts}
        return outputs

    def _calculate_loss(self, logits, ident_target, typing_target, ident_mask, typing_mask):
        """
        Calculate loss for two-stage model
        """

        ident_loss = typing_loss = 0

        ident_logits, typing_logits = logits

        if self.use_crf:
            ident_target = ident_target * ident_mask
            ident_loss = -self.crf(ident_logits, ident_target, reduction='mean', mask=ident_mask)
        else:
            ident_loss = self.ident_loss_fn(
                ident_logits.reshape(-1, self.ident_num_labels), ident_target.reshape(-1)
            )

        typing_loss = self._span_label_loss(typing_logits, typing_target, typing_mask)

        loss = ident_loss + typing_loss

        return loss

    def _span_label_loss(self, typing_logits, typing_target, typing_mask):

        num_types = typing_logits.size(-1)

        typing_target = typing_target.reshape(-1).to(torch.long)  # B*M x L
        typing_logits = typing_logits.reshape(-1, typing_logits.size(-1))
        typing_mask = typing_mask.reshape(-1).bool()

        typing_logits = typing_logits.masked_select(typing_mask.unsqueeze(-1)).reshape(
            -1, num_types
        )
        typing_target = typing_target.masked_select(typing_mask)
        typing_loss = self.typing_loss_fn(typing_logits, typing_target)
        if typing_mask.sum() == 0:
            return typing_loss.sum()
        typing_loss = typing_loss.sum() / typing_mask.sum()
        return typing_loss

    def bio2span(self, sequences, device):
        """label sequence to span format"""

        bsz = sequences.size(0)

        all_spans = [[] for _ in range(bsz)]

        for b in range(len(all_spans)):
            all_spans[b] = self._convert2span(sequences[b].tolist())

        max_num_spans = max(len(i) for i in all_spans)

        if max_num_spans == 0:
            max_num_spans = 1

        pred_mentions = [i + (max_num_spans - len(i)) * [[-1, -1]] for i in all_spans]

        pred_mentions = torch.tensor(pred_mentions).to(device)
        pred_mask = pred_mentions[..., 0] != -1

        return pred_mentions, pred_mask

    def _convert2span(self, label_list):
        span_list = []
        i = 0
        label_list = [self.ident_id_to_label[item] for item in label_list]
        while i < len(label_list):
            if label_list[i][0] == 'B' or label_list[i][0] == 'I':
                start_idx = i
                i += 1
                while i < len(label_list) and not (
                    label_list[i][0] == 'O' or label_list[i][0] == 'E'
                ):
                    i += 1
                if i < len(label_list):
                    end_idx = i if label_list[i][0] == 'E' else i - 1
                    # Looks like a good trick
                    span_list.append([start_idx, end_idx])
                    i += 1
                else:
                    span_list.append([start_idx, i - 1])

            elif label_list[i][0] == 'S' or label_list[i][0] == 'E':
                span_list.append([i, i])
                i += 1
            else:
                i += 1

        return span_list

    def decode(self, embed, ident_mask):  # noqa: D102

        # for detection

        ident_logits = self._forward_ident(embed)
        if self.use_crf:
            ident_predicts = self.crf.decode(ident_logits, mask=ident_mask).squeeze(0)
        else:
            ident_predicts = ident_logits.argmax(-1)  # B*M

        pred_mentions, pred_mask = self.bio2span(ident_predicts, ident_logits.device)

        mention_boundary = pred_mentions.transpose(-1, -2)

        typing_logits = self._forward_typing(embed, mention_boundary)
        typing_predicts = typing_logits.argmax(-1)
        typing_predicts = typing_predicts.detach().cpu().numpy()

        batch_mentions = []
        mention_list = pred_mentions.tolist()
        offset_unit = len(mention_list[0])
        for b in range(len(mention_list)):
            mentions = []
            for i, mention in enumerate(mention_list[b]):
                if mention[0] == -1:
                    try:
                        assert mention[1] == -1, breakpoint()
                    except IndexError:
                        breakpoint()
                    continue
                mentions.append(
                    (
                        mention[0],
                        mention[1] + 1,
                        self.typing_id_to_label[typing_predicts[i + b * offset_unit]],
                    )
                )
            batch_mentions.append(mentions)

        return batch_mentions
