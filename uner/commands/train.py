import argparse
import os

from modelscope.trainers import build_trainer
from modelscope.utils.config import Config

from uner.commands.subcommand import Subcommand


class Train(Subcommand):

    @classmethod
    def add_subparser(
            cls,
            parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser('train', help='train a model')
        subparser.add_argument(
            '-c', '--cfg_file', required=True, help='configuration YAML file')
        subparser.add_argument(
            '-t', '--trainer', default=None, help='trainer name')
        subparser.add_argument(
            '-cp', '--checkpoint_path', default=None, help='model checkpoint')
        subparser.add_argument(
            '--seed',
            type=int,
            default=None,
            help='random seed for everything')
        subparser.add_argument('--local_rank', type=int, default=0)

        subparser.set_defaults(func=train_model_from_args)
        return subparser


def train_model_from_args(args: argparse.Namespace):
    trainer = build_trainer_from_args(args)
    trainer.train(args.checkpoint_path)
    trainer.test()


def build_trainer_from_args(args):
    if args.trainer is not None:
        trainer_name = args.trainer
    else:
        cfg = Config.from_file(args.cfg_file)
        assert 'trainer' in cfg, 'trainer must be specified!'
        trainer_name = cfg.trainer

    kwargs = vars(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        kwargs['device'] = 'gpu'
        kwargs['launcher'] = 'pytorch'

    trainer = build_trainer(trainer_name, kwargs)
    return trainer
