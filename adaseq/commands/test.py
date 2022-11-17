# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

from modelscope.trainers import build_trainer
from modelscope.utils.config import Config

from adaseq.commands.subcommand import Subcommand


class Test(Subcommand):
    """
        usage: adaseq test [-h] -c CFG_FILE [-t TRAINER] [-cp CHECKPOINT_PATH]

        optional arguments:
          -h, --help            show this help message and exit
          -c CFG_FILE, --cfg_file CFG_FILE
                                configuration YAML file
          -t TRAINER, --trainer TRAINER
                                trainer name
          -cp CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                                model checkpoint
    """

    @classmethod
    def add_subparser(cls, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """ Add testing arguments parser """
        subparser = parser.add_parser('test', help='test with a model checkpoint')
        subparser.add_argument('-c', '--cfg_file', required=True, help='configuration YAML file')
        subparser.add_argument('-t', '--trainer', default=None, help='trainer name')
        subparser.add_argument('-cp', '--checkpoint_path', default=None, help='model checkpoint')

        subparser.set_defaults(func=test_model_from_args)
        return subparser


def test_model_from_args(args: argparse.Namespace):  # noqa
    trainer = build_trainer_from_args(args)
    trainer.test(args.checkpoint_path)


def build_trainer_from_args(args):  # noqa
    if args.trainer is not None:
        trainer_name = args.trainer
    else:
        cfg = Config.from_file(args.cfg_file)
        assert 'trainer' in cfg, 'trainer must be specified!'
        trainer_name = cfg.trainer

    kwargs = vars(args)
    trainer = build_trainer(trainer_name, kwargs)
    return trainer
