# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import torch

from adaseq.data.constant import PARTIAL_LABEL_ID
from adaseq.metainfo import Decoders

from .base import DECODERS
from .crf import CRF


@DECODERS.register_module(module_name=Decoders.partial_crf)
class PartialCRF(CRF):
    """Partial/Fuzzy Conditional random field."""

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        super().__init__(num_tags, batch_first)
        self.UNLABELED_INDEX = PARTIAL_LABEL_ID
        self.IMPOSSIBLE_SCORE = -100

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
        self._validate(emissions, tags=tags, mask=mask)

        # partial mask
        possible_tags = self._create_possible_tag_masks(self.num_tags, tags)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            possible_tags = possible_tags.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask, possible_tags)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def _create_possible_tag_masks(self, num_tags: int, tags: torch.Tensor) -> torch.Tensor:
        # process no annotation idx
        tags = tags.clone()
        no_annotation_idx = tags == self.UNLABELED_INDEX
        tags[no_annotation_idx] = 0

        # create tag masks
        masks = torch.zeros(
            tags.size(0), tags.size(1), num_tags, dtype=torch.uint8, device=tags.device
        )
        masks.scatter_(2, tags.unsqueeze(2), 1)
        masks[no_annotation_idx] = 1  # enumerate all tags if no annotation

        return masks

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.ByteTensor,
        possible_tags: torch.ByteTensor,
    ) -> torch.Tensor:
        """
        Parameters:
            emissions: (seq_length, batch_size, num_tags)
            tags: (seq_length, batch_size)
            mask: (seq_length, batch_size)
            possible_tags: (seq_length, batch_size, num_tags)
        Returns:
            scores: (batch_size)
        """

        sequence_length, batch_size, num_tags = emissions.data.shape

        mask = mask.float()
        possible_tags = possible_tags.float()

        # Start transition score and first emission
        first_possible_tag = possible_tags[0]

        alpha = self.start_transitions + emissions[0]  # (batch_size, num_tags)
        alpha[(first_possible_tag == 0)] = self.IMPOSSIBLE_SCORE

        for i in range(1, sequence_length):
            current_possible_tags = possible_tags[i - 1]
            next_possible_tags = possible_tags[i]  # (batch_size, num_tags)

            # Emissions scores
            emissions_score = emissions[i].clone()
            emissions_score[(next_possible_tags == 0)] = self.IMPOSSIBLE_SCORE
            emissions_score = emissions_score.view(batch_size, 1, num_tags)

            # Transition scores
            transition_scores = (
                self.transitions.view(1, num_tags, num_tags)
                .expand(batch_size, num_tags, num_tags)
                .clone()
            )
            transition_scores[(current_possible_tags == 0)] = self.IMPOSSIBLE_SCORE
            transition_scores.transpose(1, 2)[(next_possible_tags == 0)] = self.IMPOSSIBLE_SCORE

            # Broadcast alpha
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all scores
            inner = (
                broadcast_alpha + emissions_score + transition_scores
            )  # (batch_size, num_tags, num_tags)
            alpha = torch.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
                1 - mask[i]
            ).view(batch_size, 1)

        # Add end transition score
        last_tag_indexes = mask.sum(0).long() - 1
        unnamed = (
            last_tag_indexes
            + torch.arange(batch_size, device=possible_tags.device) * sequence_length
        )
        end_transitions = (
            self.end_transitions.expand(batch_size, num_tags)
            * possible_tags.transpose(0, 1).view(sequence_length * batch_size, num_tags)[unnamed]
        )
        end_transitions[(end_transitions == 0)] = self.IMPOSSIBLE_SCORE
        stops = alpha + end_transitions

        return torch.logsumexp(stops, 1)  # (batch_size,)
