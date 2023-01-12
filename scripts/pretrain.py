# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import sys
import warnings

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from adaseq.commands.pretrain import pretrain_model  # noqa # isort:skip

warnings.filterwarnings('ignore')


def main(args):
    """train a model from args"""
    pretrain_model(
        config_path=args.config_path,
        work_dir=args.work_dir,
        run_name=args.run_name,
        seed=args.seed,
        force=args.force,
        device=args.device,
        local_rank=args.local_rank,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument(
        '-c', '--config_path', required=True, type=str, help='configuration YAML file'
    )
    parser.add_argument(
        '-w',
        '--work_dir',
        type=str,
        default=None,
        help='directory to save experiment logs and checkpoints',
    )
    parser.add_argument('-n', '--run_name', type=str, default=None, help='trial name')
    parser.add_argument('-d', '--device', type=str, default='gpu', help='device name')
    parser.add_argument(
        '-f', '--force', default=None, help='overwrite the output directory if it exists.'
    )
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, help='model checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='random seed for everything')
    parser.add_argument('--local_rank', type=str, default='0')

    args = parser.parse_args()
    main(args)
