import argparse
import glob
import os
import re
import subprocess
from itertools import product
from typing import Optional

import numpy as np


def tune(args):
    from modelscope.utils.config import Config
    assert args.cfg_file is not None, 'cfg_file must be specified!'
    config = Config.from_file(args.cfg_file)
    assert 'experiment' in config
    assert 'exp_name' in config.experiment

    env_config = Config.from_file(args.config_env)
    if 'python_interpreter' not in env_config:
        env_config['python_interpreter'] = 'python'
    if 'python_file' not in env_config:
        env_config['python_file'] = 'scripts.train'
    assert 'trainer' in env_config

    print('===== experiment meta config =====')
    print(config.to_dict())

    # expand configs
    all_configs = expand_config(config.to_dict())

    op_gen = input(f'use it to generate {len(all_configs)} '
                   'config files? (y/n): ').lower()
    if op_gen in ['y', 'yes']:
        print('generating configs...')
    else:
        exit(0)

    config_path = os.path.join(config.experiment.exp_dir,
                               config.experiment.exp_name, 'configs')

    os.makedirs(config_path, exist_ok=True)

    for idx, c in enumerate(all_configs):
        config_file_path = os.path.join(config_path, f'config.{idx}.yaml')
        Config(c).dump(config_file_path)

    # select gpus
    print('===== listing gpu info ===== ')
    subprocess.run(['nvidia-smi'])

    gpu = input('select gpus (e.g. 0 or 0,1 or 0,1,3,4): ').lower()
    gpus = None
    pattern = re.compile(r'(\d|,)*\d$')
    if re.match(pattern, gpu):
        gpus = gpu.split(',')
    else:
        print('wrong gpu input!')
        exit(0)

    print('selected gpus: {}'.format(gpus))
    n_gpu = len(gpus)

    # generate scripts
    config_files = glob.glob(os.path.join(config_path, '*.yaml'))
    splitted_files = np.array_split(config_files, n_gpu)

    scripts_path = os.path.join(config.experiment.exp_dir,
                                config.experiment.exp_name, 'scripts')

    env_settings = ''
    for e in env_config.get('envs', []):
        env_settings += f'{e}\n'

    os.makedirs(scripts_path, exist_ok=True)
    all_script_files = []
    assert len(splitted_files) == len(gpus)
    for i, split in enumerate(splitted_files):
        script_file = os.path.join(scripts_path, '{}.sh'.format(i))
        all_script_files.append(script_file)
        with open(script_file, 'w') as f:
            output = ''
            output += env_settings
            for config_file in split:
                output += 'CUDA_VISIBLE_DEVICES={gpu_id} {python_interpreter} ' \
                    '-m {python_file} -t {trainer} -c {config_file} \n'.format(
                        gpu_id=gpus[i],
                        python_interpreter=env_config.python_interpreter,
                        python_file=env_config.python_file,
                        trainer=env_config.trainer,
                        config_file=config_file)
            f.write(output)

    op_run = input('run scripts ? (y/n): ').lower()
    if op_run in ['y', 'yes']:
        print('start running scripts...')
    else:
        exit(0)

    for f in all_script_files:
        subprocess.run('nohup sh {} > {} &'.format(f, args.to), shell=True)

    print('===== running =====')


def expand_config(config):
    flattened_config = flatten_config(config)
    keys = [item[0] for item in flattened_config]
    values = [
        item[1] if isinstance(item[1], list) else [item[1]]
        for item in flattened_config
    ]
    configs = []
    for single_values in product(*values):
        assert len(keys) == len(single_values)
        data = list(zip(keys, single_values))
        configs.append(create_config(data))
    return configs


def flatten_config(obj, path=[]):
    if isinstance(obj, (str, int, float)):
        return [(path, obj)]
    elif isinstance(obj, dict):
        ret = []
        for k, v in obj.items():
            ret.extend(flatten_config(v, path + [k]))
        return ret
    elif isinstance(obj, list):
        if all(isinstance(x, (str, int, float)) for x in obj):
            return [(path, obj)]
        else:
            ret = []
            for i, x in enumerate(obj):
                ret.extend(flatten_config(x, path + [i]))
            return ret
    else:
        raise ValueError(
            'Unsupported value type: {}'.format(type(obj)),
            'Only the following value types are supported: str, int, float')


def create_config(data):
    config = {}
    for path, v in data:
        temp_dict = config
        for i, k in enumerate(path):
            if isinstance(temp_dict, list):
                temp_dict += [{} for _ in range(k + 1 - len(temp_dict))]
            elif k not in temp_dict:
                if i < len(path) - 1 and isinstance(path[i + 1], int):
                    temp_dict[k] = []
                else:
                    temp_dict[k] = {}
            if i < len(path) - 1:
                temp_dict = temp_dict[k]
            else:
                temp_dict[k] = v
    return config


def collect(args):
    pass


def kill():
    subprocess.run(
        'kill_cnt=$((`ps -ef | grep $USER | grep -v "grep" | grep "sh experiments" |'
        'wc -l` + `ps -ef | grep $USER | grep -v "grep" | grep "scripts.train" | wc -l`));'
        'ps -ef | grep $USER | grep -v "grep" | grep "sh experiments";'
        'ps -ef | grep $USER | grep -v "grep" | grep "scripts.train";'
        'ps -ef | grep $USER | grep -v "grep" | grep "sh experiments" | tr -s " " | cut -d " " -f 2 | xargs kill;'
        'ps -ef | grep $USER | grep -v "grep" | grep "scripts.train" | tr -s " " | cut -d " " -f 2 | xargs kill;'
        'echo "$kill_cnt processes killed."',
        shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('grid_search.py')
    parser.add_argument(
        'mode',
        choices=['tune', 'collect', 'kill'],
        help='[tune, collect, kill]')
    parser.add_argument('-c', '--cfg_file', help='configuration YAML file')
    parser.add_argument(
        '-c_env',
        '--config_env',
        help='configuration YAML file for environment',
        default='examples/grid_search/env.yaml')
    parser.add_argument(
        '-to', '--to', help='stdout and stderr to', default='/dev/null')
    args = parser.parse_args()

    if args.mode == 'tune':
        tune(args)
    elif args.mode == 'collect':
        collect(args)
    else:
        kill()
