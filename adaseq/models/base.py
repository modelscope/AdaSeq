# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import torch
import transformers
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.base.base_torch_model import TorchModel as MsModel
from modelscope.models.builder import build_model
from modelscope.utils.checkpoint import (
    save_checkpoint,
    save_configuration,
    save_pretrained,
)
from modelscope.utils.config import Config, ConfigDict
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from packaging import version

logger = logging.getLogger(__name__)


class Model(MsModel, ABC):
    """
    The model base class
    """

    pipeline = None

    def __init__(self, model_dir: str = None, **kwargs):
        super().__init__()
        MsModel.__init__(self, model_dir=model_dir, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        def new_init(self, init=cls.__init__, *args, **kwargs):
            init(self, *args, **kwargs)
            self.post_init()

        cls.__init__ = new_init

    def post_init(self):
        """Run something after __init__ in subclass

        All derived model instances will try to load checkpoint from model_dir after __init__.
        Useful when initializing pipeline from model_id.
        """
        self.load_model_ckpt()

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:  # noqa: D102
        # use `nn.Module.__call__` rather than `MsModel.__call__`
        return self.postprocess(super()._call_impl(*args, **kwargs))

    def load_model_ckpt(self, model_dir: str = None) -> None:
        """Try to load model checkpoint from model_dir/pytorch_model.bin"""
        if model_dir is None:
            model_dir = self.model_dir
        if model_dir is not None:
            model_ckpt = os.path.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE)
            state_dict = torch.load(model_ckpt, map_location=torch.device('cpu'))
            compatible_position_ids(
                state_dict, 'embedder.transformer_model.embeddings.position_ids'
            )
            self.load_state_dict(state_dict)

    @abstractmethod
    def forward(
        self, tokens: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Run the forward pass for a model.

        Args:
            tokens (Dict[str, Any]): inputs to the model
            meta (Dict[str, Any]): raw inputs
            **kwargs: other arguments

        Returns:
            outputs (Dict[str, Any]): outputs from the model
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg_dict_or_path: Optional[Union[str, Dict, Config]] = None, **kwargs):
        """Instantiate a model from config dict or path.

        Args:
            cfg_dict_or_path (Optional[Union[str, Dict, Config]]): config dict or path
            **kwargs: other arguments

        Returns:
            model (:obj:`Model`): a model instance
        """

        if isinstance(cfg_dict_or_path, str) and osp.isfile(cfg_dict_or_path):
            cfg = Config.from_file(cfg_dict_or_path)
        elif isinstance(cfg_dict_or_path, (dict, Config)):
            cfg = cfg_dict_or_path
        else:
            cfg = dict(model=dict(), task=None)

        task: str = cfg['task']
        if 'task' in kwargs:
            task = kwargs.pop('task')
        model_config: Dict = cfg['model']

        if 'type' in kwargs:
            model_config['type'] = kwargs.pop('type')

        if 'type' not in model_config:
            raise ValueError(
                'Please pass a correct cfg dict, which should be a reachable file or a dict.'
            )

        model = build_model(model_config, task_name=task, default_args=kwargs)
        cfg['framework'] = 'pytorch'
        cfg['model'].update(kwargs)  # type: ignore
        setattr(model, 'cfg', cfg)  # follow `MsModel.save_pretrained`
        return model

    def save_pretrained(
        self,
        target_folder: Union[str, os.PathLike],
        save_checkpoint_names: str = ModelFile.TORCH_MODEL_BIN_FILE,
        save_function: Callable = save_checkpoint,
        config: Optional[dict] = None,
        save_config_function: Callable = save_configuration,
        with_meta: bool = False,
        **kwargs,
    ) -> None:
        """save the pretrained model, its configuration and other related files to a directory,
            so that it can be re-loaded

        Args:
            target_folder (Union[str, os.PathLike]):
            Directory to which to save. Will be created if it doesn't exist.

            save_checkpoint_names (Union[str, List[str]]):
            The checkpoint names to be saved in the target_folder

            save_function (Callable, optional):
            The function to use to save the state dictionary.

            config (Optional[dict], optional):
            The config for the configuration.json, might not be identical with model.config

            save_config_function (Callble, optional):
            The function to use to save the configuration.
        """
        if config is None and hasattr(self, 'cfg'):
            config = self.cfg

        # save pytorch_model.bin and model related files
        save_pretrained(
            self, target_folder, save_checkpoint_names, save_function, with_meta=with_meta, **kwargs
        )

        if config is not None:
            # config modification
            config['plugins'] = ['adaseq']

            if self.pipeline is not None:
                config['pipeline'] = {'type': self.pipeline}

            if (
                'preprocessor' in config
                and 'label_to_id' not in config['preprocessor']
                and 'model' in config
                and 'id_to_label' in config['model']
            ):
                config['preprocessor']['label_to_id'] = {
                    v: int(k) for k, v in config['model']['id_to_label'].items()
                }

            # save configuration.json
            save_config_function(target_folder, config)

        # embedder configuration
        if not osp.isfile(osp.join(target_folder, 'config.json')):
            try:
                self.embedder.transformer_model.config.save_pretrained(target_folder)
            except Exception as e:
                logger.warning(f'embedder config.json not saved! {str(e)}')

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        revision: Optional[str] = DEFAULT_MODEL_REVISION,
        cfg_dict: Config = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Instantiate a model from local directory or remote model repo. Note
        that when loading from remote, the model revision can be specified.

        Args:
            model_name_or_path(str): A model dir or a model id to be loaded
            revision(str, `optional`): The revision used when the model_name_or_path is
                a model id of the remote hub. default `master`.
            cfg_dict(Config, `optional`): An optional model config. If provided, it will replace
                the config read out of the `model_name_or_path`
            device(str, `optional`): The device to load the model.
            **kwargs:
                task(str, `optional`): The `Tasks` enumeration value to replace the task value
                read out of config in the `model_name_or_path`. This is useful when the model to be loaded is not
                equal to the model saved.
                For example, load a `backbone` into a `text-classification` model.
                Other kwargs will be directly fed into the `model` key, to replace the default configs.
        Returns:
            A model instance.

        Examples:
            >>> from modelscope.models import Model
            >>> Model.from_pretrained('damo/nlp_structbert_backbone_base_std', task='text-classification')
        """
        prefetched = kwargs.get('model_prefetched')
        if prefetched is not None:
            kwargs.pop('model_prefetched')

        if osp.exists(model_name_or_path):
            local_model_dir = model_name_or_path
        else:
            if prefetched is True:
                raise RuntimeError('Expecting model is pre-fetched locally, but is not found.')
            local_model_dir = snapshot_download(model_name_or_path, revision)
        logger.info(f'initialize model from {local_model_dir}')
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg = Config.from_file(osp.join(local_model_dir, ModelFile.CONFIGURATION))

        model_cfg = cfg.model
        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type

        for k, v in kwargs.items():
            model_cfg[k] = v
        if device is not None:
            model_cfg.device = device

        model = build_model(model_cfg, default_args=kwargs)
        model.model_dir = local_model_dir

        # dynamically add pipeline info to model for pipeline inference
        if hasattr(cfg, 'pipeline'):
            model.pipeline = cfg.pipeline
        else:
            model.pipeline = ConfigDict(type=cfg.task)

        # cfg['framework'] = 'pytorch'
        for key in ('cfg', 'config'):
            if not hasattr(model, key):
                setattr(model, key, cfg)

        model.name = model_name_or_path
        return model


def compatible_position_ids(state_dict, position_id_key):
    """Transformers no longer expect position_ids after transformers==4.31
       https://github.com/huggingface/transformers/pull/24505
    Args:
        position_id_key (str): position_ids key,
            such as(encoder.embeddings.position_ids)
    """
    transformer_version = version.parse('.'.join(transformers.__version__.split('.')[:2]))
    if transformer_version >= version.parse('4.31.0'):
        del state_dict[position_id_key]
