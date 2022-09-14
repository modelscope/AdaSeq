import torch
import torch.nn as nn


class WordDropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(WordDropout, self).__init__()
        assert 0.0 <= dropout_rate < 1.0, '0.0 <= dropout rate < 1.0 must be satisfied!'
        self.dropout_rate = dropout_rate

    def forward(self, inputs):
        if not self.training or not self.dropout_rate:
            return inputs

        m = inputs.data.new(inputs.size(0), 1, 1).bernoulli_(1.0 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(inputs)
        return inputs * mask
