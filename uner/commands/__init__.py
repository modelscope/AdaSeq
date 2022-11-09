import argparse
from typing import Optional, Tuple

from uner import __version__
from uner.commands.test import Test
from uner.commands.train import Train


def parse_args(
    prog: Optional[str] = None
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(help='commands')
    Train.add_subparser(subparsers)
    Test.add_subparser(subparsers)

    args = parser.parse_args()

    return parser, args


def main(prog: Optional[str] = None) -> None:
    parser, args = parse_args(prog)

    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
