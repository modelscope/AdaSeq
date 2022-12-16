# Copyright (c) Alibaba, Inc. and its affiliates.
from os import path as osp
from typing import Dict, Union

import torch.nn as nn
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg

ENCODERS = Registry('encoders')


def build_encoder(cfg: ConfigDict, default_args: dict = None):
    """Build encoder from config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for encoder object
        default_args (dict): default initialization arguments

    Returns:
        encoder (:obj:`Encoder`): an encoder instance
    """
    return build_from_cfg(cfg, ENCODERS, group_key='default', default_args=default_args)


class Encoder(nn.Module):
    """
    The encoder base class for encoding embeddings to features.
    """

    def get_input_dim(self) -> int:
        """
        Get the input feature dim.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Get the output feature dim.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg_dict_or_path: Union[str, Dict, Config] = None, **kwargs):
        """Build encoder instance from config"""
        if isinstance(cfg_dict_or_path, str) and osp.isfile(cfg_dict_or_path):
            cfg = Config.from_file(cfg_dict_or_path).model.encoder
        elif isinstance(cfg_dict_or_path, (dict, Config)):
            cfg = cfg_dict_or_path
        else:
            cfg = {}

        if 'type' not in cfg:
            cfg['type'] = None

        if 'type' in kwargs:
            cfg['type'] = kwargs.pop('type')

        if cfg['type'] is not None and cfg['type'] in ENCODERS.modules['default']:
            return build_encoder(cfg, default_args=kwargs)
        else:
            raise ValueError('Unsupported encoder type: %s', cfg['type'])
