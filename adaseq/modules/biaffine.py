# Copyright (c) Alibaba, Inc. and its affiliates.

import torch


class Biaffine(torch.nn.Module):
    """Biaffine Attention"""

    def __init__(self, in_features: int, out_features: int, bias=(True, True)):
        super().__init__()
        self.in_features = in_features  # mlp_arc_size / mlp_label_size
        self.out_features = out_features  # 1 / rel_size
        self.bias = bias

        # arc: mlp_size
        # label: mlp_size + 1
        self.linear_input_size = in_features + bias[0]
        # arc: mlp_size * 1
        # label: (mlp_size + 1) * rel_size
        self.linear_output_size = out_features * (in_features + bias[1])

        self.linear = torch.nn.Linear(
            in_features=self.linear_input_size, out_features=self.linear_output_size, bias=False
        )
        # self.reset_params()

    # def reset_params(self):
    #     nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input1, input2):  # noqa: D102
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
            # dim1 += 1
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)
            # dim2 += 1

        # (bz, len1, dim1+1) -> (bz, len1, linear_output_size)
        affine = self.linear(input1)

        # (bz, len1 * self.out_features, dim2)
        affine = affine.reshape(batch_size, len1 * self.out_features, -1)

        # (bz, len1 * out_features, dim2) * (bz, dim2, len2)
        # -> (bz, len1 * out_features, len2) -> (bz, len2, len1 * out_features)
        biaffine = torch.bmm(affine, input2.transpose(1, 2)).transpose(1, 2).contiguous()

        # (bz, len2, len1, out_features)    # out_features: 1 or rel_size
        biaffine = biaffine.reshape((batch_size, len2, len1, -1)).squeeze(-1)

        return biaffine
