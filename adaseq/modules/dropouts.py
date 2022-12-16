# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn


class WordDropout(nn.Module):
    """Word-level Dropout module

    During training, randomly zeroes some of the elements of the input tensor at word level
    with probability `dropout_rate` using samples from a Bernoulli distribution.

    Args:
        dropout_rate (float): dropout rate for each word
    """

    def __init__(self, dropout_rate: float = 0.1):
        super(WordDropout, self).__init__()
        assert 0.0 <= dropout_rate < 1.0, '0.0 <= dropout rate < 1.0 must be satisfied!'
        self.dropout_rate = dropout_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Dropout the input tensor at word level

        Args:
            inputs (torch.Tensor): input tensor

        Returns:
            outputs (torch.Tensor): output tensor of the same shape as input
        """
        if not self.training or not self.dropout_rate:
            return inputs

        mask = inputs.new_empty(*inputs.shape[:2], 1, requires_grad=False).bernoulli_(
            1.0 - self.dropout_rate
        )
        mask = mask.expand_as(inputs)
        return inputs * mask
