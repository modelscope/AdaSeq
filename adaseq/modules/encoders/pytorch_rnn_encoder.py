# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) AI2 AllenNLP. Licensed under the Apache License, Version 2.0.

from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from adaseq.metainfo import Encoders
from adaseq.utils.checks import ConfigurationError

from .base import ENCODERS, Encoder


class _PytorchRnnWrapper(Encoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a `get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from `get_output_dim`.

    Note that we *require* you to pass a binary mask of shape (batch_size, sequence_length)
    when you call this module, to avoid subtle bugs around masking.  If you already have a
    `PackedSequence` you can pass `None` as the second parameter.
    """

    def __init__(self, module: nn.RNNBase) -> None:
        super().__init__()
        self._module = module
        if not self._module.batch_first:
            raise ConfigurationError('Our encoder semantics assumes batch is always first!')

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        return self._module.hidden_size * (1 + int(self._module.bidirectional))

    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.BoolTensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        inputs: `torch.Tensor`
            Shape: [batch_size, max_length, input_dim].
        mask: `torch.BoolTensor`
            Shape: [batch_size, max_length].

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, max_length, output_dim]`.
        """
        # If you already have a `PackedSequence`, you can pass `None` mask.
        if mask is None:
            return self._module(inputs)[0]

        sequence_lengths = mask.sum(-1)

        # First count how many sequences are empty.
        num_empty = torch.sum(~mask[:, 0]).int().item()
        if num_empty > 0:
            raise RuntimeError(f'Got {num_empty} empty sequences, please check your data.')

        # Now create a PackedSequence.
        packed_sequence_input = pack_padded_sequence(
            inputs, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Actually call the module on the sorted PackedSequence.
        packed_sequence_output, _ = self._module(packed_sequence_input)

        # unpack and restore the batch
        feature, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        return feature


@ENCODERS.register_module(module_name=Encoders.gru_encoder)
class GruEncoder(_PytorchRnnWrapper):
    """
    Registered as a `Encoder` with name "gru".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)


@ENCODERS.register_module(module_name=Encoders.lstm_encoder)
class LstmEncoder(_PytorchRnnWrapper):
    """
    Registered as a `Encoder` with name "lstm".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)


@ENCODERS.register_module(module_name=Encoders.rnn_encoder)
class RnnEncoder(_PytorchRnnWrapper):
    """
    Registered as a `Encoder` with name "rnn".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)
