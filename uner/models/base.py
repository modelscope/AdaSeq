import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import torch.nn as nn
from modelscope.utils.config import Config
from modelscope.models.builder import build_model

from uner.utils.common_utils import has_keys


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def _instantiate(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def from_config(cls,
                    cfg_dict_or_path: Optional[Union[str, Dict, Config]] = None,
                    **kwargs):

        if isinstance(cfg_dict_or_path, str) and osp.isfile(cfg_dict_or_path):
            cfg = Config.from_file(cfg_dict_or_path).model
        elif isinstance(cfg_dict_or_path, (dict, Config)):
            cfg = cfg_dict_or_path
        else:
            cfg = {}

        if 'type' in kwargs:
            cfg['type'] = kwargs.pop('type')
        
        if 'type' not in cfg:
            raise ValueError('Please pass a correct cfg dict, which should be a reachable file or a dict.')

        model = build_model(cfg, default_args=kwargs)
        return model

