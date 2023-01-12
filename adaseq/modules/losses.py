from typing import Optional

import torch
import torch.nn as nn
from torch.nn import BCELoss, BCEWithLogitsLoss


class WBCEWithLogitsLoss:
    """Weighed BCE loss, multiply the loss of positive examples with a scaler"""

    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight
        self.loss = BCEWithLogitsLoss(reduction='none')

    def __call__(self, y_pred, y_true):  # noqa: D102
        loss = self.loss(y_pred, y_true.float())

        if self.pos_weight != 1:
            weights_pr = torch.ones_like(y_true).float()  # B x C
            weights_pr[y_true > 0] = self.pos_weight
            loss = (loss * weights_pr).mean()
        else:
            loss = loss.mean()

        return loss


class WBCEWithLogitsLossUFET:
    """Weighed BCE loss, multiply the loss of positive examples with a scaler,
    Apply the trick in Ultra-Fine Entity Typing with Weak Supervision from a Masked Language Model"""

    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight
        self.loss = BCEWithLogitsLoss(reduction='none')

    def __call__(self, y_pred, y_true):  # noqa: D102
        loss = self.loss(y_pred, y_true.float())
        B, L = y_true.shape
        coarse = y_true[:, :9].sum(-1) > 0
        fine = y_true[:, 9:130].sum(-1) > 0
        ultra = y_true[:, 130:].sum(-1) > 0
        a = coarse.view(-1, 1) * torch.ones_like(y_true[:, :9])
        b = fine.view(-1, 1) * torch.ones_like(y_true[:, 9:130])
        c = ultra.view(-1, 1) * torch.ones_like(y_true[:, 130:])
        weight_cfu = torch.cat([a, b, c], -1)
        loss = loss * weight_cfu
        if self.pos_weight != 1:
            weights_pr = torch.ones_like(y_true).float()  # B x C
            weights_pr[y_true > 0] = self.pos_weight
            loss = (loss * weights_pr).mean()
        else:
            loss = loss.mean()

        return loss


class PartialBCELoss(BCELoss):
    """PartialBCELoss"""

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction='none',
        total_num_labels=0,
        negative_sampling=0.3,
    ):
        super(PartialBCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.num_labels = total_num_labels
        self.negative_sampling = negative_sampling

    def forward(self, input, target):  # noqa
        return self.partial_binary_cross_entropy(input, target)

    def partial_binary_cross_entropy(self, input, target):  # noqa
        # input: B*M x L
        # target: B*M x L
        assert input.shape[0] == target.shape[0] and input.shape[1] == target.shape[1]
        if self.negative_sampling > 0:
            # breakpoint()
            negative_nums = int(self.num_labels * self.negative_sampling)
            sel_negative_ids = torch.randint(0, self.num_labels, (negative_nums, 1)).squeeze()
            sel_negatives = torch.tensor([0] * self.num_labels).to(target.device)
            sel_negatives[sel_negative_ids] = 1
            loss = -(
                target * torch.log(input)
                + (1 - target) * torch.log(1 - input) * sel_negatives.unsqueeze(0)
            )
            # loss = loss / negative_nums
        else:
            loss = -target * torch.log(input)
        # loss = loss / target.shape[1]
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.sum() / (target.shape[0])
        return loss


class PULearningLoss(nn.Module):
    """PU Learning Loss.
    https://aclanthology.org/P19-1231.pdf
    https://aclanthology.org/2020.findings-emnlp.60.pdf
    """

    def __init__(self, num_tags, label2id: dict, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.label2id = label2id
        self.batch_first = batch_first
        self.new_tag = {item for item in label2id if item != 'O'}
        self.prior = {item: 0.05 for item in label2id if item != 'O'}
        self.balance = {item: 0.8 for item in label2id if item != 'O'}
        # self.new_tag = {'B-SPAN', 'I-SPAN', 'E-SPAN', 'S-SPAN'}
        # self.prior   = {'B-SPAN': 0.05, 'I-SPAN': 0.05,'E-SPAN': 0.05, 'S-SPAN': 0.05}
        # self.balance = {'B-SPAN': 0.8, 'I-SPAN': 0.8, 'E-SPAN': 0.8, 'S-SPAN': 0.8}
        self.gamma = 1.0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def masked_sum(self, logits, mask):
        """sum with mask"""

        if mask.sum() == 0:
            return torch.sum(logits * mask)
        return torch.sum(logits * mask) / mask.sum()

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()

        # calculate pRisk
        one_hot_targets = nn.functional.one_hot(tags, self.num_tags)
        unit_risk = torch.sum(torch.pow(emissions - one_hot_targets, 2), dim=-1)
        pRisk = 0.0
        for tag in self.new_tag:
            tag_id = self.label2id[tag]
            pMask = mask * (tags == tag_id)
            risk = self.masked_sum(unit_risk, pMask)
            pRisk += self.balance[tag] * risk

        # calculate nRisk: nRisk = uRisk - self.prior * (1 - pRisk)
        one_hot_targets = nn.functional.one_hot(torch.zeros_like(tags), self.num_tags)
        unit_risk = torch.sum(torch.pow(emissions - one_hot_targets, 2), dim=-1)
        _pRisk = 0.0
        for tag in self.new_tag:
            tag_id = self.label2id[tag]
            pMask = mask * (tags == tag_id)
            risk = self.masked_sum(unit_risk, pMask)
            _pRisk += self.prior[tag] * risk

        uMask = mask * (tags == 0)  # self.label2id['O'] = 0
        uRisk = self.masked_sum(unit_risk, uMask)

        nRisk = uRisk - _pRisk

        # calculate final_risk
        if nRisk < 0:
            final_risk = -nRisk
        else:
            final_risk = self.gamma * pRisk + nRisk

        return final_risk
