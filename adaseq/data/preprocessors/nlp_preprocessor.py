# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Union

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Models
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.hub import get_model_type
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast

from adaseq.metainfo import Preprocessors
from adaseq.utils.data_utils import gen_label2id


@PREPROCESSORS.register_module(module_name=Preprocessors.nlp_preprocessor)
class NLPPreprocessor(Preprocessor):
    """common pre-process operations for NLP tasks"""

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(**kwargs)

        self.max_length = kwargs.pop('max_length', 512)
        self.add_cls_sep = kwargs.pop('add_cls_sep', True)
        self.return_tokens_or_text = kwargs.pop('return_tokens_or_text', True)
        self.return_attention_mask = kwargs.pop('return_attention_mask', True)
        self.return_emission_mask = kwargs.pop('return_emission_mask', False)
        self.return_offsets_mapping = kwargs.pop('return_offsets_mapping', False)
        self.return_original_view = kwargs.pop('return_original_view', False)
        use_fast = kwargs.pop('use_fast', True)
        revision = kwargs.pop('revision', None)
        self.tokenizer = self.build_tokenizer(model_dir, use_fast, revision)

    def build_tokenizer(self, model_dir, use_fast: bool = False, revision=None):
        """build tokenizer from `transformers`."""
        tokenizer = BertTokenizerFast if use_fast else BertTokenizer
        if 'word2vec' in model_dir:
            return tokenizer.from_pretrained(model_dir)
        elif 'nezha' in model_dir:
            return tokenizer.from_pretrained(model_dir)
        try:
            return AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
        except OSError:
            if not os.path.exists(model_dir):
                model_dir = snapshot_download(model_dir, revision)
            # code borrowed from modelscope/preprocessors/nlp/nlp_base.py
            model_type = get_model_type(model_dir)
            if model_type in (Models.structbert, Models.gpt3, Models.palm, Models.plug):
                if use_fast:
                    # from modelscope.models.nlp.structbert import SbertTokenizerFast
                    pass
                from modelscope.models.nlp.structbert import SbertTokenizer

                return SbertTokenizer.from_pretrained(model_dir)
            elif model_type == Models.veco:
                from modelscope.models.nlp.veco import VecoTokenizer, VecoTokenizerFast

                tokenizer = VecoTokenizerFast if use_fast else VecoTokenizer
                return tokenizer.from_pretrained(model_dir)
            elif model_type == Models.deberta_v2:
                from modelscope.models.nlp.deberta_v2 import (
                    DebertaV2Tokenizer,
                    DebertaV2TokenizerFast,
                )

                tokenizer = DebertaV2TokenizerFast if use_fast else DebertaV2Tokenizer
                return tokenizer.from_pretrained(model_dir)
            else:
                raise ValueError('Unsupported tokenizer')

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """encode one instance, it could be a text str, a list of tokens for a dict"""
        if isinstance(data, str):
            data = {'text': data}
        if isinstance(data, List):
            data = {'tokens': data}
        if 'tokens' in data:
            output = self.encode_tokens(data)
            if self.return_original_view:
                output.update(self.encode_tokens_origin_view(data))
        elif 'text' in data:
            output = self.encode_text(data)
        else:
            raise ValueError('Data sample must have "text" or "tokens" field!')
        return output

    def encode_tokens(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """conver token to ids, add some mask."""
        tokens = data['tokens']
        mask = data.get('mask', [True] * len(tokens))
        input_ids = []
        emission_mask = []
        offset_mapping = []
        for offset, (token, token_mask) in enumerate(zip(tokens, mask)):
            subtoken_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(subtoken_ids) == 0:
                subtoken_ids = [self.tokenizer.unk_token_id]
            input_ids.extend(subtoken_ids)
            emission_mask.extend([token_mask] + [False] * (len(subtoken_ids) - 1))
            offset_mapping.extend(
                [(offset, offset + 1)] + [(offset + 1, offset + 1)] * (len(subtoken_ids) - 1)
            )
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[: self.max_length - 2]
            emission_mask = emission_mask[: self.max_length - 2]
            offset_mapping = offset_mapping[: self.max_length - 2]
        if self.add_cls_sep:
            input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
            emission_mask = [False] + emission_mask + [False]
            offset_mapping = [(0, 0)] + offset_mapping + [(0, 0)]
        attention_mask = [1] * len(input_ids)

        output = {
            'input_ids': input_ids,
        }
        if self.return_tokens_or_text:
            output['tokens'] = tokens
        if self.return_attention_mask:
            output['attention_mask'] = attention_mask
        if self.return_emission_mask:
            output['emission_mask'] = emission_mask
        if self.return_offsets_mapping:
            output['offset_mapping'] = offset_mapping
        return output

    def encode_tokens_origin_view(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """encode tokens when using multi-view model."""
        tokens = data['tokens']
        mask = data.get('mask', [True] * len(tokens))

        # remove the padded context
        sent_length = sum(mask)
        origin_tokens = tokens[:sent_length]
        origin_mask = mask[:sent_length]

        input_ids = []
        emission_mask = []
        offset_mapping = []
        for offset, (token, token_mask) in enumerate(zip(origin_tokens, origin_mask)):
            subtoken_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(subtoken_ids) == 0:
                subtoken_ids = [self.tokenizer.unk_token_id]
            input_ids.extend(subtoken_ids)
            offset_mapping.extend(
                [(offset, offset + 1)] + [(offset + 1, offset + 1)] * (len(subtoken_ids) - 1)
            )
            emission_mask.extend([token_mask] + [False] * (len(subtoken_ids) - 1))
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[: self.max_length - 2]
            offset_mapping = offset_mapping[: self.max_length - 2]
            emission_mask = emission_mask[: self.max_length - 2]
        if self.add_cls_sep:
            input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
            offset_mapping = [(0, 0)] + offset_mapping + [(0, 0)]
            emission_mask = [False] + emission_mask + [False]
        attention_mask = [1] * len(input_ids)

        output = {
            'origin_input_ids': input_ids,
        }
        if self.return_tokens_or_text:
            output['origin_tokens'] = origin_tokens
        if self.return_attention_mask:
            output['origin_attention_mask'] = attention_mask
        if self.return_emission_mask:
            output['origin_emission_mask'] = emission_mask
        if self.return_offsets_mapping:
            output['origin_offset_mapping'] = offset_mapping
        return output

    def encode_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """encode 'text' to ids"""
        raise NotImplementedError

    def map_label_to_id(
        self, labels: List[str] = None, label2id: Dict[str, int] = None
    ) -> Dict[str, int]:
        """conver labels to ids"""
        if label2id is not None:
            return label2id
        elif labels is not None:
            return self._label2id(labels)
        else:
            raise ValueError('labels or label2id is needed.')

    def _label2id(self, labels: List[str]) -> Dict[str, int]:
        return gen_label2id(labels)
