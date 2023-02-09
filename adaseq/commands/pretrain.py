# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Optional, Union

import yaml
from modelscope.utils.config import Config
from modelscope.utils.torch_utils import set_random_seed

from adaseq.data.data_collators.base import build_data_collator
from adaseq.data.dataset_manager import DatasetManager
from adaseq.data.preprocessors.nlp_preprocessor import build_preprocessor
from adaseq.metainfo import Trainers
from adaseq.training import build_trainer
from adaseq.utils.checks import ConfigurationError
from adaseq.utils.common_utils import create_datetime_str
from adaseq.utils.file_utils import is_empty_dir
from adaseq.utils.logging import prepare_logging


def pretrain_model(
    config_path_or_dict: Union[str, dict],
    work_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
    force: bool = False,
    device: str = 'gpu',
    local_rank: str = '0',
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Train a model from config file or dict.
    You can manually call this function in a python script for debugging.
    """
    if isinstance(config_path_or_dict, str):
        config = Config.from_file(config_path_or_dict)
    else:
        config = Config(config_path_or_dict)

    # create work_dir
    if work_dir is None:
        work_dir = os.path.join(
            config.safe_get('experiment.exp_dir', 'experiments/'),
            config.safe_get('experiment.exp_name', 'unknown/'),
            run_name or create_datetime_str(),
        )
    if os.path.exists(work_dir) and force:
        shutil.rmtree(work_dir, ignore_errors=True)
    if os.path.exists(work_dir) and not is_empty_dir(work_dir):
        raise ConfigurationError(f'`work_dir` ({work_dir}) already exists and is not empty.')
    os.makedirs(work_dir, exist_ok=True)

    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    prepare_logging(work_dir, int(local_rank), world_size)

    # Get seed from the comand line args first.
    if seed is None:
        # if not given, try to get one from config file
        seed = config.safe_get('experiment.seed', 42)  # 42 by default
        if seed < 0:
            raise ConfigurationError(f'random seed must be greater than 0, got: {seed}')
    else:
        config['experiment']['seed'] = seed
    set_random_seed(seed)

    trainer = build_pretrainer_from_partial_objects(
        config,
        work_dir=work_dir,
        seed=seed,
        device=device,
        local_rank=local_rank,
    )
    trainer.train(checkpoint_path)
    trainer.test()


def build_pretrainer_from_partial_objects(config, work_dir, **kwargs):
    """
    Entrypoint of build the trainer from `config` by modelscope.
    In this method, we will build the `DatasetManager` first, then use the
    counted or loaded `labels` to build the`Preprocessor`.
    The the`Preprocessor` will provide the final `id_to_label` mapping,
    which is a required argument of all `AdaSeq` models, we update it to the
    `model` section of `config` and dump the updated `config` to the `work_dir`.

    Args:

    work_dir (`str`): required
        The created directionary to save all produced files in training.
    config (`Config`): required
        The `Config` of this trial.
    """
    # build datasets via `DatasetManager`
    dm = DatasetManager.from_config(task=config.task, **config.dataset)
    # build preprocessor with config and labels
    preprocessor = build_preprocessor(config.preprocessor, labels=dm.labels)

    if 'lr_scheduler' not in config.train:  # default constant lr.
        config.train['lr_scheduler'] = dict(type='constant')

    # Finally, get `id_to_label` for model.
    config.model.typing_id_to_label = preprocessor.typing_id_to_label
    config.model.ident_id_to_label = preprocessor.ident_id_to_label
    # Dump config to work_dir and reload.
    new_config_path = os.path.join(work_dir, 'config.yaml')
    with open(new_config_path, mode='w', encoding='utf8') as file:
        yaml.dump(config.to_dict(), file, allow_unicode=True)

    # build `DataCollator` from config and tokenizer.
    collator_config = config.data_collator
    if isinstance(collator_config, str):
        collator_config = dict(type=collator_config)
    data_collator = build_data_collator(preprocessor.tokenizer, collator_config)

    trainer = build_trainer(
        config.safe_get('train.trainer', Trainers.default_trainer),
        cfg_file=new_config_path,
        work_dir=work_dir,
        dataset_manager=dm,
        data_collator=data_collator,
        preprocessor=preprocessor,
        **kwargs,
    )
    return trainer
