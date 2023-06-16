# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as func

from adaseq.metainfo import Encoders

from .base import ENCODERS, Encoder


@ENCODERS.register_module(module_name=Encoders.cnn_encoder)
class CnnEncoder(Encoder):
    """Turn token embedding sequenece to a single vector."""

    def __init__(
        self,
        input_size: int,
        channels: int = 128,
        dropout: float = 0.0,
        dilation: Optional[List[int]] = None,
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )
        if dilation is None:
            dilation = [1, 2, 3]
        self.input_dim = input_size
        self.output_dim = channels * len(dilation)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d)
                for d in dilation
            ]
        )

    def forward(self, x, **kwargs):  # noqa: D102
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = func.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs

    def get_input_dim(self) -> int:  # noqa: D102
        return self.input_dim

    def get_output_dim(self) -> int:  # noqa: D102
        return self.output_dim
