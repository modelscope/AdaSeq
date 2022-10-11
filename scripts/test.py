import argparse
import warnings

from modelscope.trainers import build_trainer

import uner

warnings.filterwarnings('ignore')


def main(args):
    trainer = build_trainer(args.trainer_name, vars(args))
    trainer.test(args.checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument(
        '-c', '--cfg_file', required=True, help='configuration YAML file')
    parser.add_argument(
        '-t', '--trainer_name', required=True, help='trainer name')
    parser.add_argument(
        '-cp', '--checkpoint_path', default=None, help='model checkpoint')
    args = parser.parse_args()

    main(args)
