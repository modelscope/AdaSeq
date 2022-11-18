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
    """test a model from args"""
    trainer = build_trainer(args)
    trainer.test(args.checkpoint_path)


def build_trainer(args):
    """build a trainer from args"""
    if args.trainer is not None:
        trainer_name = args.trainer
    else:
        cfg = Config.from_file(args.cfg_file)
        assert 'trainer' in cfg, 'trainer must be specified!'
        trainer_name = cfg.trainer

    kwargs = vars(args)
    trainer = ms_build_trainer(trainer_name, kwargs)
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('-c', '--cfg_file', required=True, help='configuration YAML file')
    parser.add_argument('-t', '--trainer', default=None, help='trainer name')
    parser.add_argument('-cp', '--checkpoint_path', required=True, help='model checkpoint')
    args = parser.parse_args()

    main(args)
