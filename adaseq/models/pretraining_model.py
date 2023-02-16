# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from modelscope.models.builder import MODELS

from adaseq.data.tokenizer import build_tokenizer
from adaseq.metainfo import Models, Tasks
from adaseq.models.base import Model
from adaseq.models.utils import bio2span
from adaseq.modules.decoders import CRF
from adaseq.modules.embedders import Embedder, TransformerEmbedder
from adaseq.modules.encoders import SpanEncoder
from adaseq.modules.util import get_tokens_mask


@MODELS.register_module(Tasks.entity_typing, module_name=Models.pretraining_model)
@MODELS.register_module(Tasks.named_entity_recognition, module_name=Models.pretraining_model)
class PretrainingModel(Model):
    """Pretraining model

    Args:
        num_labels (int): number of labels
        embedder (Union[embedder, str], `optional`): embedder used in the model.
            It can be an `Embedder` instance or an embedder config file or an embedder config dict.
        span_encoder_method (str): concat
        word_dropout (float, `optional`): word dropout rate, default `0.0`.
        class_threshold (float, `optional`): classification threshold default. '0.5'
        use_biaffine (bool, `optional`): whether to use biaffine for span representation, default `False`.
        **kwargs: other arguments
    """

    def __init__(
        self,
        typing_id_to_label: Dict[int, str],
        ident_id_to_label: Dict[int, str],
        embedder: Union[Embedder, str],
        span_encoder_method: str = 'concat',
        word_dropout: Optional[float] = 0.0,
        class_threshold: float = 0.5,
        use_biaffine: bool = False,
        **kwargs
    ):
        super(PretrainingModel, self).__init__(**kwargs)
        self.ident_num_labels = len(ident_id_to_label)
        self.typing_num_labels = len(typing_id_to_label)
        self.ident_id_to_label = {int(k): v for k, v in ident_id_to_label.items()}
        self.typing_id_to_label = {int(k): v for k, v in typing_id_to_label.items()}
        if isinstance(embedder, Embedder):
            self.embedder = embedder
        else:
            self.embedder = Embedder.from_config(embedder)
            # WARNING: hard code
            self.tokenizer = build_tokenizer('bert-base-chinese')
        assert isinstance(self.embedder, TransformerEmbedder)

        self.lstm = nn.LSTM(768, 384, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm_dropout = torch.nn.Dropout(0.3)

        self.use_dropout = word_dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(word_dropout)

        self.class_threshold = class_threshold

        self.use_biaffine = use_biaffine
        self.negative_sampling_ratio = kwargs.get('negative_sampling_ratio', 0.5)
        self.typing_penalize = kwargs.get('typing_penalize', 1.0)

        self.split_space_ident = nn.Linear(
            self.embedder.get_output_dim(), self.embedder.get_output_dim()
        )
        self.ident_linear = nn.Linear(self.embedder.get_output_dim(), self.ident_num_labels)
        self.crf = CRF(self.ident_num_labels, batch_first=True)

        self.span_encoder = SpanEncoder(
            self.embedder.get_output_dim(),
            span_encoder_method,
            use_biaffine=self.use_biaffine,
            **kwargs
        )
        self.split_space_typing = nn.Linear(
            self.embedder.get_output_dim(), self.embedder.get_output_dim()
        )
        self.typing_linear = nn.Linear(self.span_encoder.output_dim, self.typing_num_labels)
        self.split_space_prompt = nn.Linear(
            self.embedder.get_output_dim(), self.embedder.get_output_dim()
        )
        self.prompt_linear = nn.Linear(self.embedder.get_output_dim(), self.ident_num_labels)
        self.prompt_crf = CRF(self.ident_num_labels, batch_first=True)
        self.load_model_ckpt()

    def __forward_ident(self, token_embed: torch.Tensor) -> torch.Tensor:
        ident_embed = self.split_space_ident(token_embed[:, 1:-1, :])
        ident_logits = self.ident_linear(ident_embed)
        return ident_logits

    def __forward_typing(
        self, token_embed: torch.Tensor, mention_boundary: torch.LongTensor
    ) -> torch.Tensor:
        typing_embed = self.split_space_typing(token_embed[:, 1:-1, :])
        # B*M x K
        span_reprs = self.span_encoder(typing_embed, mention_boundary)
        typing_logits = self.typing_linear(span_reprs)
        return typing_logits

    def __forward_prompt_ner(
        self, prompt_input_ids: torch.Tensor, offsets: torch.Tensor, prompt_input_mask: torch.Tensor
    ) -> torch.Tensor:
        subtoken_embed = self.embedder.encode(prompt_input_ids, prompt_input_mask)
        token_embed = self.embedder.reconstruct(subtoken_embed, offsets)
        ident_embed = self.split_space_prompt(token_embed[:, 1:-1, :])
        ident_logits = self.prompt_linear(ident_embed)
        return ident_logits

    def forward(  # noqa: D102
        self,
        tokens: Dict[str, Any],
        prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_input_mask: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # TODO docstring
        if not self.training and prompt_input_ids is None:
            predicts = self._forward_two_stage_ner(tokens)
            outputs = {'predicts': predicts}
            return outputs
        elif not self.training and prompt_input_ids is not None:
            predicts = self._forward_zero_shot_ner(tokens, prompt_input_ids, prompt_input_mask)
            outputs = {'predicts': predicts}
            return outputs

    def _forward_two_stage_ner(  # noqa: D102
        self,
        tokens: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # TODO docstring
        # B x M -> B*M x 1
        subtoken_embed = self.embedder.encode(tokens['input_ids'], tokens['attention_mask'])
        token_embed = self.embedder.reconstruct(subtoken_embed, tokens['offsets'])
        lstm_output, _ = self.lstm(token_embed)
        token_embed = self.lstm_dropout(lstm_output)
        ident_logits = self.__forward_ident(token_embed)
        ident_mask = get_tokens_mask(tokens, ident_logits.size(1))
        ident_predicts = self.crf.decode(ident_logits, mask=ident_mask).squeeze(0)
        pred_mentions, pred_mask = bio2span(
            ident_predicts, ident_logits.device, self.ident_id_to_label
        )

        mention_boundary = pred_mentions.transpose(-1, -2)

        typing_logits = self.__forward_typing(token_embed, mention_boundary)
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

    def _forward_zero_shot_ner(  # noqa: D102
        self,
        tokens: Dict[str, Any],
        prompt_input_ids: torch.Tensor,
        prompt_input_mask: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # TODO docstring
        # B x M -> B*M x 1
        prompt_logits = self.__forward_prompt_ner(
            prompt_input_ids, tokens['offsets'], prompt_input_mask
        )
        label_mask = get_tokens_mask(tokens, prompt_logits.size(1))
        prompt_predicts = self.prompt_crf.decode(prompt_logits, mask=label_mask).squeeze(0)
        pred_mentions, _ = bio2span(prompt_predicts, prompt_logits.device, self.ident_id_to_label)

        batch_mentions = []
        mention_list = pred_mentions.tolist()
        for b in range(len(mention_list)):
            mentions = []
            for i, mention in enumerate(mention_list[b]):
                if mention[0] == -1:
                    try:
                        assert mention[1] == -1, breakpoint()
                    except IndexError:
                        breakpoint()
                    continue
                mentions.append((mention[0], mention[1] + 1, 'SPAN'))
            batch_mentions.append(mentions)
        return batch_mentions
