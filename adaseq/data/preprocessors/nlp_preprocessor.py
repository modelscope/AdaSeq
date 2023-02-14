# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.builder import build_preprocessor as ms_build_preprocessor
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Fields, ModeKeys

from adaseq.data.tokenizer import build_tokenizer
from adaseq.metainfo import Preprocessors

logger = logging.getLogger(__name__)


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.nlp_preprocessor)
class NLPPreprocessor(Preprocessor):
    """
    Some common pre-process operations for NLP tasks.

    Args:
        model_dir (str): pre-trained model name or path.
        is_word2vec (bool): if True, tokens will be indexed in word2vec style,
            i.e., 1 to 1 mapping, without word pieces.
        tokenizer_kwargs (Optional[Dict[str, Any]]): some arguments to init tokenizer
            from huggingface, modelscope or ...
        max_length (int): we will discard tokens that exceed the `max_length`.
            So please take care of this argument.
        return_offsets (bool): if `True`, compute sub-token offset mapping for the
            original sequence reconstruction in the `TransformerEncoder`.
        add_special_tokens (bool): add special tokens of pre-trained models to the
            input, it is only effective when `return_offsets==False`.
        labels (List[str]): a label list, which is used to setup a `label_to_id`
            mappging if the following `label_to_id` was not provided.
        label_to_id (Dict[str, int]): a dict maps label to index,
            such as `{'O': 0, 'B-LOC': 1, 'I-LOC': 2}`.
        return_original_view (bool): if `True`, return token_ids and other tensors
            that without padded context, only used in retrieval-augmented models.
    """

    def __init__(
        self,
        model_dir: str,
        is_word2vec: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
        return_offsets: bool = False,
        add_special_tokens: bool = True,
        labels: Optional[List[str]] = None,
        label_to_id: Optional[Dict[str, int]] = None,
        return_original_view: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if 'word2vec' in model_dir and is_word2vec is False:
            is_word2vec = True
            add_special_tokens = False
            logger.warning(
                'You are using word2vec embedder, auto set `is_word2vec = True` `add_special_tokens = False`'
            )

        self.is_word2vec = is_word2vec
        self.max_length = max_length if self.mode == ModeKeys.TRAIN else 512
        self.add_special_tokens = add_special_tokens
        self.return_offsets = return_offsets
        self.return_original_view = return_original_view
        self.tokenizer = build_tokenizer(
            model_dir, is_word2vec=is_word2vec, **(tokenizer_kwargs or {})
        )

        if label_to_id is not None:
            self.label_to_id = label_to_id
        elif labels is not None:
            self.label_to_id = self.make_label_to_id(labels)
        else:
            raise ValueError('Must have one of `labels` or `label_to_id`')
        self.id_to_label = {v: k for k, v in sorted(self.label_to_id.items(), key=lambda x: x[1])}
        # make sure they are aligned.
        assert len(self.id_to_label) not in self.id_to_label
        assert len(self.id_to_label) - 1 in self.id_to_label

    def __call__(self, data: Union[str, List, Dict]) -> Dict[str, Any]:
        """
        Encode one instance, it could be a text str, a list of tokens for a dict.

        Returns:
            Dict[str, Any]: `{'tokens': tokenized and encoded tensors, 'meta': data input}`
        """
        if isinstance(data, str):
            data = {'text': data}
        if isinstance(data, List):
            data = {'tokens': data}

        output_dict = {'meta': data}

        if 'tokens' in data:
            output_dict['tokens'] = self.encode_tokens(data['tokens'])
        elif 'text' in data:
            output_dict['tokens'] = self.encode_text(data['text'])
        else:
            raise ValueError('Data sample must have "text" or "tokens" field!')

        if self.return_original_view is not None:
            output_dict['origin_mask'] = data['mask']
            if self.return_original_view:
                # return token_ids and other tensors that without padded context,
                # only used in retrieval-augmented models.
                output_dict['origin_tokens'] = self.encode_tokens_origin_view(data)

        if self.mode == ModeKeys.INFERENCE:
            for key, value in output_dict['tokens'].items():
                output_dict['tokens'][key] = np.expand_dims(np.array(value), 0)

        return output_dict

    def encode_text(self, text: str) -> Dict[str, Any]:
        """encode `text` to ids"""
        inputs_with_offsets = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=self.return_offsets
        )
        encoded = self.encode_tokens(inputs_with_offsets.tokens())
        if self.return_offsets and self.mode == ModeKeys.INFERENCE:
            encoded['offset_mapping'] = inputs_with_offsets['offset_mapping']
        return encoded

    def encode_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Convert tokens to ids, add some mask.
        """
        if self.is_word2vec:
            return self.encode_tokens_word2vec(tokens)
        else:
            return self.encode_tokens_wordpiece(tokens)

    def encode_tokens_word2vec(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Convert tokens to ids, one by one via vocab, no word pieces.
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        encoded = {'input_ids': input_ids, 'mask': [True] * len(tokens)}
        return encoded

    def encode_tokens_wordpiece(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Convert tokens to ids by word piece tokenizer.
        """
        input_ids = []
        # the corresponding inclusive sub-token span of tokens
        offsets: List[Optional[Tuple[int, int]]] = []

        max_length = self.max_length

        if self.add_special_tokens:
            input_ids.append(self.tokenizer.cls_token_id)
            offsets.append((0, 0))

            # if `add_special_tokens`, the max_length should minus 1 for the appending `[SEP]`
            max_length -= int(self.add_special_tokens)

        for token_string in tokens:
            wordpieces = self.tokenizer.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            wp_ids = wordpieces['input_ids']

            # For tokens that don't correspond to any word pieces, we set it to [UNK].
            if len(wp_ids) == 0:
                wp_ids = [self.tokenizer.unk_token_id]

            offsets.append((len(input_ids), len(input_ids) + len(wp_ids) - 1))
            input_ids.extend(wp_ids)

            if len(input_ids) >= max_length:
                # discard sub-tokens that exceed the `max_length`
                input_ids = input_ids[:max_length]
                offsets[-1] = (offsets[-1][0], len(input_ids) - 1)
                break

        if self.add_special_tokens:
            offsets.append((len(input_ids), len(input_ids)))
            input_ids.append(self.tokenizer.sep_token_id)

        encoded = {
            'input_ids': input_ids,
            'attention_mask': [True] * len(input_ids),
            'has_special_tokens': self.add_special_tokens,
        }
        if self.return_offsets:
            encoded['mask'] = [True] * len(offsets)
            encoded['offsets'] = offsets
        return encoded

    def make_label_to_id(self, labels: List[str]) -> Dict[str, int]:
        """
        Generate `label_to_id` mapping.
        You can override this method to customize the mapping.
        NOTE: The `self.labels` could be modified in this method.
        """
        return {label: i for i, label in enumerate(labels)}

    def encode_tokens_origin_view(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """encode tokens when using retrieval-augmented multi-view model."""
        tokens = data['tokens']
        mask = data.get('mask', [True] * len(tokens))
        # remove the padded context
        origin_length = sum(mask)
        origin_tokens = tokens[:origin_length]
        return self.encode_tokens(origin_tokens)

    def save_pretrained(
        self,
        target_folder: Union[str, os.PathLike],
        config: Optional[dict] = None,
        save_config_function: Callable = None,
    ):
        """Save the preprocessor, its configuration and other related files to a directory,
            so that it can be re-loaded

        By default, this method will save the preprocessor's config with mode `inference`.

        Args:
            target_folder (Union[str, os.PathLike]):
            Directory to which to save. Will be created if it doesn't exist.

            config (Optional[dict], optional):
            The config for the configuration.json

            save_config_function (Callable): The function used to save the configuration, call this function
                after the config is updated.

        """
        super().save_pretrained(target_folder, config, save_config_function)
        # save tokenizer
        if not osp.isfile(osp.join(target_folder, 'vocab.txt')):
            self.tokenizer.save_pretrained(target_folder)


def build_preprocessor(config: ConfigDict, **kwargs) -> Preprocessor:
    """Build preprocessor from config"""
    # get `field_name` for loading modelscope preprocessors,
    # Not sure who will need this.
    field_name = config.get('field_name', Fields.nlp)
    return ms_build_preprocessor(config, field_name, kwargs)  # type: ignore
