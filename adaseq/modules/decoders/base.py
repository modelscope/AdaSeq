# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import abstractmethod
from os import path as osp
from typing import Dict, Union

import torch
import torch.nn as nn
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg
from transformers import AutoModel

from adaseq.utils.common_utils import has_keys

DECODERS = Registry('decoders')


def build_decoder(cfg: ConfigDict, default_args: dict = None):
    """Build decoder from config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for decoder object
        default_args (dict): default initialization arguments

    Returns:
        encoder (:obj:`Decoder`): an decoder instance
    """
    return build_from_cfg(cfg, DECODERS, group_key='default', default_args=default_args)


class Decoder(nn.Module):
    """
    The decoder base class for downstream tasks
    """

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def instantiate(cls, model_name_or_path=None, config=None, **kwargs):
        """Instantiate a decoder class, the __init__ method will be called by default.

        Args:
            model_name_or_path: The model_name_or_path parameter used to initialize a decoder.
            config: The config dict read from the caller's cfg_dict_or_path parameter.
            **kwargs: Extra parameters.

        Returns: A Decoder instance.
        """
        return cls(model_name_or_path=model_name_or_path, config=config, **kwargs)

    @abstractmethod
    @torch.jit.export
    def decode(self, logits, mask=None, **kwargs):
        """Decode logits"""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, model_name_or_path: str = None, cfg_dict_or_path: Union[str, Dict] = None, **kwargs
    ):
        """Build an decoder subclass.

        Args:
            model_name_or_path: The model_name_or_path parameter used to initialize a decoder.
            cfg_dict_or_path: The extra config file or the extra config dict.
            **kwargs:
                decoder_type: Same with cfg.model.decoder.type

        Returns: An Decoder instance.
        """
        assert (
            model_name_or_path is not None or cfg_dict_or_path is not None
        ), 'Either the model or the cfg information should be passed in from the parameters.'

        if cfg_dict_or_path is not None:
            if isinstance(cfg_dict_or_path, str) and osp.isfile(cfg_dict_or_path):
                cfg = Config.from_file(cfg_dict_or_path)
            elif isinstance(cfg_dict_or_path, (dict, Config)):
                cfg = cfg_dict_or_path
            else:
                raise ValueError(
                    'Please pass a correct cfg dict, which should be a reachable file or a dict.'
                )
        elif model_name_or_path is not None and osp.exists(
            osp.join(model_name_or_path, 'config.json')
        ):
            cfg = Config.from_file(osp.join(model_name_or_path, 'config.json'))
        else:
            cfg = {}

        type = None
        if 'decoder_type' in kwargs:
            type = kwargs.pop('decoder_type')
        elif has_keys(cfg, 'model', 'decoder', 'type'):
            type = cfg['model']['decoder']['type']
        elif 'model' not in cfg and 'type' in cfg:
            type = cfg['type']
        elif 'model' not in cfg and 'model_type' in cfg:
            type = cfg['model_type']

        if model_name_or_path is None and has_keys(cfg, 'model', 'decoder', 'model_name_or_path'):
            model_name_or_path = cfg['model']['decoder']['model_name_or_path']

        if type is not None and type in DECODERS.modules['default']:
            return build_decoder(
                type,
                model_name_or_path=model_name_or_path,
                config=cfg if len(cfg) > 0 else None,
                **kwargs
            )
        else:
            assert model_name_or_path is not None, (
                'Model is not found in registry, '
                'so it is considered a huggingface backbone '
                'and the model_name_or_path param should not be None'
            )
            return AutoModel.from_pretrained(model_name_or_path, **kwargs)
