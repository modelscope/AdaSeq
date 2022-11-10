import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch.nn as nn
from modelscope.models.builder import build_model
from modelscope.utils.config import Config


class Model(nn.Module, ABC):
    """
    The model base class
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the forward pass for a model.

        Args:
            inputs (Dict[str, Any]): inputs to the model
            **kwargs: other arguments

        Returns:
            outputs (Dict[str, Any]): outputs from the model
        """
        raise NotImplementedError

    @classmethod
    def _instantiate(cls, **kwargs):
        """
        Define the instantiation method of a model, default method is calling the constructor.
        """
        return cls(**kwargs)

    @classmethod
    def from_config(cls,
                    cfg_dict_or_path: Optional[Union[str, Dict,
                                                     Config]] = None,
                    **kwargs):
        """ Instantiate a model from config dict or path.

        Args:
            cfg_dict_or_path (Optional[Union[str, Dict, Config]]): config dict or path
            **kwargs: other arguments

        Returns:
            model (:obj:`Model`): a model instance
        """

        if isinstance(cfg_dict_or_path, str) and osp.isfile(cfg_dict_or_path):
            cfg = Config.from_file(cfg_dict_or_path).model
        elif isinstance(cfg_dict_or_path, (dict, Config)):
            cfg = cfg_dict_or_path
        else:
            cfg = {}

        if 'type' in kwargs:
            cfg['type'] = kwargs.pop('type')

        if 'type' not in cfg:
            raise ValueError(
                'Please pass a correct cfg dict, which should be a reachable file or a dict.'
            )

        model = build_model(cfg, default_args=kwargs)
        return model
