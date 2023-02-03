# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Callable, Dict, List, Optional, Set, Union

from modelscope.utils.config import ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg, default_group
from transformers import PreTrainedTokenizerBase

from adaseq.data.batch import DataBatch
from adaseq.metainfo import DataCollators

DATA_COLLATORS = Registry('data_collators')


def build_data_collator(
    tokenizer: PreTrainedTokenizerBase, cfg: ConfigDict, default_args: Optional[dict] = None
):
    """build data collator from config."""
    if default_args is None:
        default_args = {}
    default_args['tokenizer'] = tokenizer
    return build_from_cfg(cfg, DATA_COLLATORS, group_key=default_group, default_args=default_args)


@DATA_COLLATORS.register_module(module_name=DataCollators.data_collator_with_padding)
class DataCollatorWithPadding:
    """
    A `DataCollator` support padding some fields to same length.
    Support padding encoder related fields: input_ids, token_type_ids, mask,
    and padding other fields with `default_pad_id`.
    `no_pad_fields` will be skipped.
    """

    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self, tokenizer, default_pad_id: int = 0, no_pad_fields: Optional[Set[str]] = None, **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.default_pad_id = default_pad_id
        self.keep_fields: Set[str] = {'tokens', 'origin_tokens', 'meta'}
        self.no_pad_fields = no_pad_fields or set()

    @staticmethod
    def _get_pad_func(padding_side: str) -> Callable:
        if padding_side == 'right':

            def _pad(array, size: int, pad_value):
                return array + [pad_value] * size

        elif padding_side == 'left':

            def _pad(array, size: int, pad_value):
                return [pad_value] * size + array

        else:
            raise ValueError('Invalid padding strategy:' + str(padding_side))
        return _pad

    def padding_token(self, batch: Dict[str, Any], padding_side: str) -> Dict[str, Any]:
        """pad token related fields (hf.transformers style)"""
        _pad = self._get_pad_func(padding_side)
        batch_size = len(batch['meta'])
        for field in [f for f in batch.keys() if f.endswith('tokens')]:
            sub_field_pair = [
                ('input_ids', self.tokenizer.pad_token_id),
                ('attention_mask', False),
                ('mask', False),
            ]
            if 'token_type_ids' in batch[field][0]:
                sub_field_pair.append(('token_type_ids', self.tokenizer.pad_token_type_id))
            if 'offsets' in batch[field][0]:
                sub_field_pair.append(('offsets', (0, 0)))

            sub = 'has_special_tokens'
            try:
                padded_tokens = {sub: [batch[field][i][sub] for i in range(batch_size)]}
            except KeyError:
                padded_tokens = {}

            for sub, pad_value in sub_field_pair:
                if sub not in batch[field][0]:
                    continue
                padded_field = list()
                max_length = max(len(i[sub]) for i in batch[field])
                for i in range(batch_size):
                    difference = max_length - len(batch[field][i][sub])
                    if difference > 0:
                        padded_field.append(_pad(batch[field][i][sub], difference, pad_value))
                    else:
                        padded_field.append(batch[field][i][sub])
                padded_tokens[sub] = padded_field
            batch[field] = padded_tokens

        field = 'origin_mask'
        if field in batch:
            max_length = max(len(i) for i in batch[field])
            for i in range(batch_size):
                difference = max_length - len(batch[field][i])
                if difference > 0:
                    batch[field][i] = _pad(batch[field][i], difference, False)

        return batch

    def padding(
        self,
        batch: Dict[str, Any],
        padding_side: str,
        fields: Optional[Union[Set[str], str]] = None,
        pad_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """pad other fields."""
        pad_id = self.default_pad_id if pad_id is None else pad_id
        _pad = self._get_pad_func(padding_side)

        if fields is None:
            fields = set(batch.keys())
        elif isinstance(fields, str):
            fields = {fields}
        fields -= self.keep_fields.union(self.no_pad_fields)

        for field in fields:
            if not isinstance(batch[field][0], list):
                continue
            max_length = max(len(i) for i in batch[field])
            for i in range(len(batch[field])):
                difference = max_length - len(batch[field][i])
                if difference > 0:
                    batch[field][i] = _pad(batch[field][i], difference, pad_id)
        return batch

    def __call__(self, instances: List[Dict[str, Any]]) -> DataBatch:
        """pad list of instances to batch"""
        batch = {key: [one[key] for one in instances] for key in instances[0].keys()}
        padding_side = self.tokenizer.padding_side
        padded_batch = self.padding_token(batch, padding_side)
        padded_batch = self.padding(padded_batch, padding_side=padding_side)
        batch = DataBatch(padded_batch, self.keep_fields)
        return batch
