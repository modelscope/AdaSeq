# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import torch
import torch.nn as nn

from adaseq.metainfo import Encoders

from .base import ENCODERS, Encoder


@ENCODERS.register_module(module_name=Encoders.span_encoder)
class SpanEncoder(Encoder):
    """Turn token embedding sequenece to a single vector."""

    def __init__(
        self,
        input_dim: int,
        encode_span_method: str = 'concat',
        add_span_linear: bool = True,
        span_hidden_size: Optional[int] = None,
        use_biaffine: bool = False,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.add_span_linear = add_span_linear
        self.encode_span_method = encode_span_method
        if self.add_span_linear:
            self.span_hidden_size = input_dim
            if span_hidden_size is not None:
                self.span_hidden_size = span_hidden_size
            self.start_linear_mapping = nn.Linear(input_dim, self.span_hidden_size)
            self.end_linear_mapping = nn.Linear(input_dim, self.span_hidden_size)
        self.use_biaffine = use_biaffine
        if self.use_biaffine:
            if self.add_span_linear:
                self.output_dim = self.span_hidden_size
            else:
                self.output_dim = self.input_dim
        elif encode_span_method == 'concat':
            if self.add_span_linear:
                self.span_reprs_dim = 2 * self.span_hidden_size
                self.output_dim = self.span_reprs_dim
            else:
                self.span_reprs_dim = 2 * self.input_dim
                self.output_dim = self.span_reprs_dim

    def get_input_dim(self) -> int:  # noqa: D102
        return self.input_dim

    def get_output_dim(self) -> int:  # noqa: D102
        return self.output_dim

    def forward(self, token_embed, span_boundary):  # noqa
        # B x N x K -> select accroding to span_boundary -> B x M x K
        batch_size = span_boundary.shape[0]
        token_embed = token_embed.reshape(batch_size, -1, self.input_dim)
        span_start_idx = span_boundary[:, 0, :]
        span_end_idx = span_boundary[:, 1, :]
        span_start_embed = token_embed[torch.arange(batch_size)[:, None], span_start_idx]
        span_end_embed = token_embed[torch.arange(batch_size)[:, None], span_end_idx]
        if self.add_span_linear:
            span_start_reprs = self.start_linear_mapping(span_start_embed)
            span_end_reprs = self.end_linear_mapping(span_end_embed)
        else:
            span_start_reprs = span_start_embed
            span_end_reprs = span_end_embed
        if self.use_biaffine:
            return span_start_reprs.reshape(-1, self.span_hidden_size), span_end_reprs.reshape(
                -1, self.span_hidden_size
            )
        if self.encode_span_method == 'concat':
            span_reprs = torch.cat((span_start_reprs, span_end_reprs), -1)
        else:
            raise NotImplementedError
        span_reprs = span_reprs.reshape(-1, self.span_reprs_dim)
        return span_reprs
