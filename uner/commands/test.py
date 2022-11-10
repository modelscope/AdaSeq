import argparse

from modelscope.trainers import build_trainer
from modelscope.utils.config import Config

from uner.commands.subcommand import Subcommand


class Test(Subcommand):

    @classmethod
    def add_subparser(
            cls,
            parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(
            'test', help='test with a model checkpoint')
        subparser.add_argument(
            '-c', '--cfg_file', required=True, help='configuration YAML file')
        subparser.add_argument(
            '-t', '--trainer', default=None, help='trainer name')
        subparser.add_argument(
            '-cp', '--checkpoint_path', default=None, help='model checkpoint')

        subparser.set_defaults(func=test_model_from_args)
        return subparser


def test_model_from_args(args: argparse.Namespace):
    trainer = build_trainer_from_args(args)
    trainer.test(args.checkpoint_path)


def build_trainer_from_args(args):
    if args.trainer is not None:
        trainer_name = args.trainer
    else:
        cfg = Config.from_file(args.cfg_file)
        assert 'trainer' in cfg, 'trainer must be specified!'
        trainer_name = cfg.trainer

    kwargs = vars(args)
    trainer = build_trainer(trainer_name, kwargs)
    return trainer
