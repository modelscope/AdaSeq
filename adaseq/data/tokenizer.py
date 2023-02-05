# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.preprocessors.nlp.transformers_tokenizer import NLPTokenizer
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer

from adaseq.utils.hub_utils import get_or_download_model_dir


def build_tokenizer(
    model_name_or_path: str, use_fast: bool = True, is_word2vec: bool = False, **kwargs
) -> PreTrainedTokenizer:
    """build tokenizer from `transformers`."""

    if is_word2vec:
        tokenizer = BertTokenizerFast if use_fast else BertTokenizer
        return tokenizer.from_pretrained(model_name_or_path, **kwargs)

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast, **kwargs)

    except OSError:
        model_name_or_path = get_or_download_model_dir(model_name_or_path)
        return NLPTokenizer(model_name_or_path, use_fast=use_fast).tokenizer
