# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Models
from modelscope.utils.hub import get_model_type
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer


def build_tokenizer(
    model_name_or_path: str, use_fast: bool = True, is_word2vec: bool = False, **kwargs
) -> PreTrainedTokenizer:
    """build tokenizer from `transformers`."""
    tokenizer = BertTokenizerFast if use_fast else BertTokenizer

    if is_word2vec or 'nezha' in model_name_or_path:
        return tokenizer.from_pretrained(model_name_or_path, **kwargs)

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast, **kwargs)

    except OSError:
        if not os.path.exists(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path, **kwargs)
        # code borrowed from modelscope/preprocessors/nlp/nlp_base.py
        model_type = get_model_type(model_name_or_path)

        if model_type in (Models.structbert, Models.gpt3, Models.palm, Models.plug):
            return tokenizer.from_pretrained(model_name_or_path, **kwargs)

        elif model_type == Models.veco:
            from transformers import XLMRobertaTokenizer, XLMRobertaTokenizerFast

            tokenizer = XLMRobertaTokenizerFast if use_fast else XLMRobertaTokenizer

            return tokenizer.from_pretrained(model_name_or_path, **kwargs)

        elif model_type == Models.deberta_v2:
            from modelscope.models.nlp.deberta_v2 import (
                DebertaV2Tokenizer,
                DebertaV2TokenizerFast,
            )

            tokenizer = DebertaV2TokenizerFast if use_fast else DebertaV2Tokenizer
            return tokenizer.from_pretrained(model_name_or_path, **kwargs)

        else:
            raise ValueError('Unsupported tokenizer')
