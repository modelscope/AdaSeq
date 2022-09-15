from abc import abstractmethod
from os import path as osp
from typing import Dict, Union

import torch.nn as nn
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.registry import Registry, build_from_cfg
from transformers import AutoModel

from uner.utils.common_utils import has_keys

ENCODERS = Registry('encoders')


def build_encoder(cfg: ConfigDict,
                  task_name: str = None,
                  default_args: dict = None):
    return build_from_cfg(
        cfg, ENCODERS, group_key=task_name, default_args=default_args)


class Encoder(nn.Module):

    @classmethod
    def _instantiate(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def from_config(cls,
                    cfg_dict_or_path: Union[str, Dict, Config] = None,
                    **kwargs):
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

        if 'model_name_or_path' not in cfg:
            cfg['model_name_or_path'] = None
        if 'model_name_or_path' in kwargs:
            cfg['model_name_or_path'] = kwargs.pop('model_name_or_path')

        if cfg['type'] is not None and cfg['type'] in ENCODERS.modules[
                'default']:
            return build_encoder(cfg, default_args=kwargs)
        else:
            assert cfg['model_name_or_path'] is not None, \
                'Model is not found in registry, ' \
                'so it is considered a huggingface backbone ' \
                'and the model_name_or_path param should not be None'
            return AutoModel.from_pretrained(cfg['model_name_or_path'],
                                             **kwargs)
