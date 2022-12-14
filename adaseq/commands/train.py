# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import shutil
from typing import Optional

from modelscope.utils.config import Config
from modelscope.utils.torch_utils import set_random_seed

from adaseq.commands.subcommand import Subcommand
from adaseq.metainfo import Trainers
from adaseq.training import build_trainer
from adaseq.utils.checks import ConfigurationError
from adaseq.utils.common_utils import create_datetime_str


class Train(Subcommand):
    """
    usage: adaseq train [-h] CONFIG_PATH [-n/--run_name RUN_NAME]
            [-d/--device DEVICE] [-f/--force] [-cp CHECKPOINT_PATH]
            [--seed SEED] [--local_rank LOCAL_RANK]

    optional arguments:
      -h, --help            show this help message and exit
      -n RUN_NAME, --run_name RUN_NAME
                            a trial name to save training files.
      -d DEVICE, --device DEVICE
                            device name for PyTorch.
      -f, --force           overwrite the output directory if it exists.
      -cp CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                            model checkpoint
      --seed SEED           random seed for everything
      --local_rank LOCAL_RANK
    """

    @classmethod
    def add_subparser(cls, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """Add training arguments parser"""
        subparser = parser.add_parser('train', help='train a model')
        subparser.add_argument('config_path', type=str, help='configuration YAML file')
        subparser.add_argument('-n', '--run_name', type=str, default=None, help='trial name.')
        subparser.add_argument('-d', '--device', type=str, default='gpu', help='device name.')
        subparser.add_argument(
            '-f', '--force', default=None, help='overwrite the output directory if it exists.'
        )
        subparser.add_argument('-cp', '--checkpoint_path', default=None, help='model checkpoint')
        subparser.add_argument('--seed', type=int, default=None, help='random seed for everything')
        subparser.add_argument('--local_rank', type=str, default='0')

        subparser.set_defaults(func=train_model_from_args)
        return subparser


def train_model_from_args(args: argparse.Namespace):  # noqa: D103
    train_model(
        config_path=args.config_path,
        run_name=args.run_name,
        seed=args.seed,
        force=args.force,
        device=args.device,
        local_rank=args.local_rank,
        checkpoint_path=args.checkpoint_path,
    )


def train_model(
    config_path: str,
    run_name: Optional[str] = None,
    seed: Optional[int] = None,
    force: bool = False,
    device: str = 'cpu',
    local_rank: str = '0',
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Train a model from config file.
    You can mannualy call this function in a python script for debugging.
    """
    config = Config.from_file(config_path)

    # create work_dir
    work_dir = os.path.join(
        config.safe_get('experiment.exp_dir', 'experiments/'),
        config.safe_get('experiment.exp_name', 'unknown/'),
        run_name or create_datetime_str(),
    )
    if os.path.exists(work_dir) and force:
        shutil.rmtree(work_dir)
    if os.path.exists(work_dir) and os.listdir(work_dir):
        raise ConfigurationError(f'`work_dir` ({work_dir}) already exists and is not empty.')
    os.makedirs(work_dir, exist_ok=True)

    # Get seed from the comand line args first.
    if seed is None:
        # if not given, try to get one from config file
        seed = config.safe_get('experiment.seed', 42)  # 42 by default
        if seed < 0:
            raise ConfigurationError(f'random seed must be greater than 0, got: {seed}')
    set_random_seed(seed)

    trainer = build_trainer(
        config.safe_get('train.trainer', Trainers.default_trainer),
        config,
        work_dir=work_dir,
        seed=seed,
        device=device,
        local_rank=local_rank,
    )
    trainer.train(checkpoint_path)
    trainer.test()
