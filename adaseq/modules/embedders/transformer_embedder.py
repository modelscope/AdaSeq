# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) AI2 AllenNLP. Licensed under the Apache License, Version 2.0.
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from modelscope.models import Model as MsModel
from transformers import AutoConfig, AutoModel, XLNetConfig

from adaseq.metainfo import Embedders
from adaseq.models.base import Model
from adaseq.modules import util
from adaseq.modules.scalar_mix import ScalarMix

from .base import EMBEDDERS, Embedder


# Code partially borrowed from https://github.com/allenai/allennlp/blob/HEAD/
# allennlp/modules/token_embedders/pretrained_transformer_embedder.py
@EMBEDDERS.register_module(module_name=Embedders.transformer_embedder)
class TransformerEmbedder(Embedder):
    """
    Uses a pretrained model from `transformers` as a `Embedder`.

    # Parameters

    model_name_or_path : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    drop_special_tokens:  `bool` (default = `True`)
        if `True`, drop the hidden states of special tokens (currently [CLS], [SEP]).
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder. Some transformers naturally act
        as embedders such as BERT. However, other models consist of encoder and decoder, in which case we just
        want to use the encoder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training. If this is `False`, the
        transformer weights are not updated during training.
    eval_mode: `bool`, optional (default = `False`)
        If this is `True`, the model is always set to evaluation mode (e.g., the dropout is disabled and the
        batch normalization layer statistics are not updated). If this is `False`, such dropout and batch
        normalization layers are only set to evaluation mode when when the model is evaluating on development
        or test data.
    load_weights: `bool`, optional (default = `True`)
        Whether to load the pretrained weights. If you're loading your model/predictor from an AllenNLP archive
        it usually makes sense to set this to `False` (via the `overrides` parameter)
        to avoid unnecessarily caching and loading the original pretrained weights,
        since the archive will already contain all of the weights needed.
    scalar_mix: `Dict[str, Any]`, optional (default = `None`)
        When `None` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings.
        If pass a kwargs dict, a scalar mix of all of the layers is used.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with additional arguments for `get_transformer`.
    sub_token_mode: `str`, (default= `first`)
        If `sub_token_mode` is set to `first`, return first sub-token representation as word-level representation.
        If `sub_token_mode` is set to `last`, return last sub-token representation as word-level representation.
        If `sub_token_mode` is set to `avg`, return average of all the sub-tokens
        representation as word-level representation.
        If `sub_token_mode` is not specified it defaults to `avg`
        If invalid `sub_token_mode` is provided, throw `ConfigurationError`
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        drop_special_tokens: bool = True,
        sub_module: Optional[str] = None,
        train_parameters: bool = True,
        eval_mode: bool = False,
        load_weights: bool = True,
        scalar_mix: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: Optional[bool] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        sub_token_mode: str = 'first',
    ) -> None:
        super().__init__()
        self.drop_special_tokens = drop_special_tokens
        self.sub_token_mode = sub_token_mode

        self.transformer_model, self.from_hf = get_transformer(
            model_name_or_path,
            load_weights=load_weights,
            **(transformer_kwargs or {}),
        )

        if self.from_hf:
            if gradient_checkpointing is not None:
                self.transformer_model.config.update(
                    {'gradient_checkpointing': gradient_checkpointing}
                )

        self.config = self.transformer_model.config

        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if scalar_mix:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers, **scalar_mix)
            self.config.output_hidden_states = True

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self.transformer_model.eval()

        if isinstance(self.config, XLNetConfig):
            self._number_of_token_type_embeddings = 3  # XLNet has 3 type ids
        elif hasattr(self.config, 'type_vocab_size'):
            self._number_of_token_type_embeddings = self.config.type_vocab_size
        else:
            self._number_of_token_type_embeddings = 0

    def train(self, mode: bool = True):  # noqa: D102
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == 'transformer_model':
                module.eval()
            else:
                module.train(mode)
        return self

    def get_output_dim(self):  # noqa: D102
        return self.output_dim

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        has_special_tokens: Optional[torch.BoolTensor] = None,
        offsets: Optional[torch.LongTensor] = None,
        mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        input_ids: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens or num_wordpieces].
        attention_mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens or num_wordpieces].
        token_type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_orig_tokens or num_wordpieces].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in input_ids. `input_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces or num_orig_tokens, embedding_size]`.
        """
        # first encode sub-token level representations
        encoded = self.encode(input_ids, attention_mask, token_type_ids)  # type: ignore
        if offsets is not None:
            # then reconstruct token-level ones by offsets
            if 'pieces2word' in kwargs:
                encoded = self.reconstruct(encoded, offsets, kwargs['pieces2word'])
            else:
                encoded = self.reconstruct(encoded, offsets)

        if has_special_tokens is not None:
            if self.drop_special_tokens and has_special_tokens.bool()[0]:
                encoded = encoded[:, 1:-1]  # So far, we only consider [CLS] and [SEP]
        return encoded

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        input_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces]`.
        attention_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        token_type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces]`.

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, embedding_size]`.

        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if token_type_ids is not None:
            max_type_id = token_type_ids.max()
            if max_type_id == 0:
                token_type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings:
                    raise ValueError('Found type ids too large for the chosen transformer model.')
                assert input_ids.shape == token_type_ids.shape

        assert attention_mask is not None
        # Shape: [batch_size, num_wordpieces, embedding_size],

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {'input_ids': input_ids, 'attention_mask': attention_mask.float()}
        if token_type_ids is not None:
            parameters['token_type_ids'] = token_type_ids

        transformer_output = self.transformer_model(**parameters)
        if self._scalar_mix is not None:
            # The hidden states will also include the embedding layer, which we don't
            # include in the scalar mix. Hence the `[1:]` slicing.
            hidden_states = transformer_output.hidden_states[1:]
            embeddings = self._scalar_mix(hidden_states)
        else:
            embeddings = transformer_output.last_hidden_state

        return embeddings

    def reconstruct(
        self,
        embeddings: torch.Tensor,
        offsets: torch.LongTensor,
        pieces2word: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        # Parameters

        input_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in input_ids. `input_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """

        # If "sub_token_mode" is set to "first", return the first sub-token embedding
        if self.sub_token_mode == 'first':
            # Select first sub-token embeddings from span embeddings
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            orig_embeddings = util.batched_index_select(embeddings, offsets[..., 0])

        # If "sub_token_mode" is set to "last", return the last sub-token embedding
        elif self.sub_token_mode == 'last':
            # Select last sub-token embeddings from span embeddings
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            orig_embeddings = util.batched_index_select(embeddings, offsets[..., 1])

        # If "sub_token_mode" is set to "avg", return the average of embeddings of all sub-tokens of a word
        elif self.sub_token_mode == 'avg':
            # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
            # span_mask: (batch_size, num_orig_tokens, max_span_length)
            span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)

            span_mask = span_mask.unsqueeze(-1)

            # Shape: (batch_size, num_orig_tokens, max_span_length, embedding_size)
            span_embeddings *= span_mask  # zero out paddings

            # Sum over embeddings of all sub-tokens of a word
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            span_embeddings_sum = span_embeddings.sum(2)

            # Shape (batch_size, num_orig_tokens)
            span_embeddings_len = span_mask.sum(2)

            # Find the average of sub-tokens embeddings by dividing `span_embedding_sum` by `span_embedding_len`
            # Shape: (batch_size, num_orig_tokens, embedding_size)
            orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

            # All the places where the span length is zero, write in zeros.
            orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        # If "sub_token_mode" is set to "w2max", return the max embedding of all sub-tokens of a word in mode of w2ner.
        elif self.sub_token_mode == 'w2max':
            min_value = torch.min(embeddings).item()
            length = pieces2word.size(1)
            bert_embeds = embeddings.contiguous().unsqueeze(1).expand(-1, length, -1, -1)
            bert_embeds = torch.masked_fill(bert_embeds, pieces2word.eq(0).unsqueeze(-1), min_value)
            orig_embeddings, _ = torch.max(bert_embeds, dim=2)

        # If invalid "sub_token_mode" is provided, throw error
        else:
            raise ValueError(f"Do not recognise 'sub_token_mode' {self.sub_token_mode}")

        return orig_embeddings


def get_transformer(
    model_name_or_path: str,
    load_weights: bool = True,
    source: Optional[str] = None,
    **kwargs,
) -> Tuple[transformers.PreTrainedModel, bool]:
    """
    Returns a transformer model and a flag of whether comes from huggingface.

    # Parameters

    model_name_or_path : `str`
        The name of the transformer, for example `"bert-base-cased"`
    load_weights : `bool`, optional (default = `True`)
        If set to `False`, no weights will be loaded. This is helpful when you only
        want to initialize the architecture, like when you've already fine-tuned a model
        and are going to load the weights from a state dict elsewhere.
        Only effective when loading model from huggingface.
    source : `str`, optional (default = `None`),
        if `source == 'huggingface'`, only try to load from huggingface.
        `source == 'modelscope'` is similar.
        By default, we will try huggingface first, and then modelscope, if both failed,
        TODO
    kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with additional arguments for `XxxxModel.from_pretrained`.

    # Returns
        Tuple[transformers.PreTrainedModel, bool], the transformer and a flag, which
        is `True` when successfully loaded from huggingface (`False` for modelscope)
    """
    from requests.exceptions import HTTPError

    from adaseq.utils.checks import ConfigurationError

    if isinstance(source, str):
        if source.lower() == 'hugginface':
            return get_hf_transformer(model_name_or_path, load_weights, **kwargs), True

        elif source.lower() == 'modelscope':
            return get_ms_transformer(model_name_or_path, **kwargs), False

        else:
            raise ConfigurationError(f'Unsupported transformer source: {source}')

    elif source is None:
        hf_e, ms_e = None, None

        try:
            return get_hf_transformer(model_name_or_path, load_weights, **kwargs), True
        except OSError as e:
            hf_e = e

        try:
            return get_ms_transformer(model_name_or_path, **kwargs), False
        except HTTPError as e:
            ms_e = e

        message = 'Try loading from huggingface and modelscope failed \n\n'
        message += 'huggingface:\n' + str(hf_e)
        message += '\n\nmodelscope:\n' + str(ms_e)
        raise RuntimeError(message)

    else:
        raise ConfigurationError(f'Unsupported transformer source: {source}')


def get_hf_transformer(
    model_name_or_path: str,
    load_weights: bool = True,
    **kwargs,
) -> transformers.PreTrainedModel:
    """see `get_transformer`."""
    if load_weights:
        transformer = AutoModel.from_pretrained(model_name_or_path, **kwargs)
    else:
        transformer = AutoModel.from_config(
            AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        )
    return transformer


def get_ms_transformer(
    model_name_or_path: str,
    **kwargs,
) -> transformers.PreTrainedModel:
    """see `get_transformer`."""
    from adaseq.utils.checks import ConfigurationError

    try:
        transformer = MsModel.from_pretrained(model_name_or_path, task='backbone', **kwargs)
    except KeyError:
        transformer = MsModel.from_pretrained(model_name_or_path, **kwargs)

        try:
            from modelscope.models.nlp.task_models.task_model import EncoderModel

            if isinstance(transformer, EncoderModel):
                transformer = transformer.encoder
            else:
                raise ConfigurationError(f'Unsupported non-backbone embedder: {model_name_or_path}')

        except (ImportError, ConfigurationError):
            model_type = transformer.cfg.model.type
            if isinstance(transformer, Model):
                transformer = transformer.embedder.transformer_model
            elif model_type == 'transformer-crf':
                transformer = transformer.model.encoder
            else:
                raise ConfigurationError(f'Unsupported non-backbone embedder: {model_name_or_path}')

    return transformer
