# Copyright (c) Alibaba, Inc. and its affiliates.
from os import path as osp
from typing import Dict, Union

import torch.nn as nn
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg

from adaseq.metainfo import Embedders

EMBEDDERS = Registry('embedders')


def build_embedder(cfg: ConfigDict, default_args: dict = None):
    """Build embedder from config dict

    Args:
        cfg (:obj:`ConfigDict`): config dict for embedder object
        default_args (dict): default initialization arguments

    Returns:
        embedder (:obj:`Embedder`): an embedder instance
    """
    return build_from_cfg(cfg, EMBEDDERS, group_key='default', default_args=default_args)


class Embedder(nn.Module):
    """
    The embedder base class for encoding input_ids to hidden-states
    """

    def get_output_dim(self) -> int:
        """
        Get the output embedding dim.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg_dict_or_path: Union[str, Dict, Config] = None, **kwargs) -> 'Embedder':
        """Build embedder instance from config"""
        if isinstance(cfg_dict_or_path, str) and osp.isfile(cfg_dict_or_path):
            cfg = Config.from_file(cfg_dict_or_path).model.embedder
        elif isinstance(cfg_dict_or_path, (dict, Config)):
            cfg = cfg_dict_or_path
        else:
            cfg = {}

        if 'type' not in cfg:
            if 'model_name_or_path' in cfg:
                cfg['type'] = Embedders.transformer_embedder
            else:
                cfg['type'] = None
        if 'type' in kwargs:
            cfg['type'] = kwargs.pop('type')

        if 'model_name_or_path' not in cfg:
            cfg['model_name_or_path'] = None
        if 'model_name_or_path' in kwargs:
            cfg['model_name_or_path'] = kwargs.pop('model_name_or_path')
            if cfg['type'] is None:
                cfg['type'] = Embedders.transformer_embedder

        if cfg['type'] is not None and cfg['type'] in EMBEDDERS.modules['default']:
            return build_embedder(cfg, default_args=kwargs)
        else:
            raise ValueError('Unsupported embedder type: %s', cfg['type'])
