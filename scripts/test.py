# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import sys
import warnings

parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_folder)

from adaseq.commands.test import test_model  # noqa: E402 isort:skip

warnings.filterwarnings('ignore')


def main(args):
    """test a model from args"""
    test_model(args.work_dir, args.device, args.checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test with a model checkpoint')
    parser.add_argument('-w', '--work_dir', required=True, help='configuration YAML file')
    parser.add_argument('-d', '--device', default='gpu', help='device name')
    parser.add_argument('-cp', '--checkpoint_path', default=None, help='model checkpoint')
    args = parser.parse_args()
    main(args)
