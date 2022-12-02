# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import sys
import warnings

from modelscope.trainers import build_trainer as ms_build_trainer
from modelscope.utils.config import Config

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

import adaseq  # noqa # isort:skip

warnings.filterwarnings('ignore')


def main(args):
    """train a model from args"""
    trainer = build_trainer(args)
    trainer.train(args.checkpoint_path)
    trainer.test()


def build_trainer(args):
    """build a trainer from args"""
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

    trainer = ms_build_trainer(trainer_name, kwargs)
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('-c', '--cfg_file', required=True, help='configuration YAML file')
    parser.add_argument('-t', '--trainer', default=None, help='trainer name')
    parser.add_argument('-cp', '--checkpoint_path', default=None, help='model checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='random seed for everything')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    main(args)
