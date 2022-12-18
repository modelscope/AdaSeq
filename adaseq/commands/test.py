# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
from typing import Optional

from modelscope.utils.config import Config

from adaseq.commands.subcommand import Subcommand
from adaseq.data.data_collators.base import build_data_collator
from adaseq.data.dataset_manager import DatasetManager
from adaseq.data.preprocessors.nlp_preprocessor import build_preprocessor
from adaseq.metainfo import Trainers
from adaseq.training import build_trainer
from adaseq.utils.checks import ConfigurationError
from adaseq.utils.file_utils import is_empty_dir


class Test(Subcommand):
    """
    usage: adaseq test [-h] -w WORK_DIR [-d DEVICE] [-ckpt CHECKPOINT_PATH]

    optional arguments:
      -h, --help            show this help message and exit
      -w WORK_DIR, --work_dir WORK_DIR
                            directory to load config and checkpoint
      -d DEVICE, --device DEVICE
                            device name
      -ckpt CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                            model checkpoint
    """

    @classmethod
    def add_subparser(cls, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """Add testing arguments parser"""
        subparser = parser.add_parser('test', help='test with a model checkpoint')
        subparser.add_argument(
            '-w', '--work_dir', required=True, help='directory to load config and checkpoint'
        )
        subparser.add_argument('-d', '--device', default='gpu', help='device name')
        subparser.add_argument('-ckpt', '--checkpoint_path', default=None, help='model checkpoint')

        subparser.set_defaults(func=test_model_from_args)
        return subparser


def test_model_from_args(args: argparse.Namespace):  # noqa: D103
    test_model(
        work_dir=args.work_dir,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
    )


def test_model(
    work_dir: str,
    device: str = 'gpu',
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Train a model from config file.
    You can mannualy call this function in a python script for debugging.
    """
    config = Config.from_file(os.path.join(work_dir, 'config.yaml'))
    checkpoint_path = checkpoint_path or os.path.join(work_dir, 'best_model.pth')

    if not os.path.exists(work_dir) and not is_empty_dir(work_dir):
        raise ConfigurationError(f'`work_dir` ({work_dir}) do not exists or is not empty.')

    # build datasets via `DatasetManager`
    dm = DatasetManager.from_config(task=config.task, **config.dataset)
    # build preprocessor with config and labels
    preprocessor = build_preprocessor(config.preprocessor, labels=dm.labels)

    # Finally, get `id_to_label` for model.
    assert config.model.id_to_label == preprocessor.id_to_label

    # build `DataCollator` from config and tokenizer.
    collator_config = config.data_collator
    if isinstance(collator_config, str):
        collator_config = dict(type=collator_config)
    data_collator = build_data_collator(preprocessor.tokenizer, collator_config)

    trainer = build_trainer(
        config.safe_get('train.trainer', Trainers.default_trainer),
        cfg_file=config.filename,
        seed=config.safe_get('experiment.seed', 42),
        device=device,
        work_dir=work_dir,
        dataset_manager=dm,
        data_collator=data_collator,
        preprocessor=preprocessor,
    )
    trainer.test(checkpoint_path)
