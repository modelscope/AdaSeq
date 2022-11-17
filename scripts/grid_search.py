# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import glob
import os
import re
import subprocess
from collections import OrderedDict
from itertools import product

import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def tune(args):
    """ hyperparameter tuning via grid search

    This function will launch experiments with multiple hyperparameter settings in parallel.
    To use this function, modify the configuration file by changing the hyperparameters you want to tune to LIST.
    for example:
    ```
      optimizer:
        type: AdamW
        lr:
          - 5.0e-5
          - 2.0e-5
        crf_lr:
          - 5.0e-1
          - 5.0e-2
    ```

    usage: python scripts/grid_search.py tune -c ${cfg_file} [-c_env ${env_file} -to ${log_file}]
    """
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
    all_configs = _expand_config(config.to_dict())

    op_gen = 'y' if args.y else input(f'use it to generate {len(all_configs)} ' 'config files? (y/n): ').lower()
    if op_gen in ['y', 'yes']:
        print('generating configs...')
    else:
        exit(0)

    config_path = os.path.join(config.experiment.exp_dir, config.experiment.exp_name, 'configs')

    os.makedirs(config_path, exist_ok=True)

    for idx, c in enumerate(all_configs):
        config_file_path = os.path.join(config_path, f'config.{idx}.yaml')
        Config(c).dump(config_file_path)

    # select gpus
    print('===== listing gpu info ===== ')
    subprocess.run(['nvidia-smi'])

    gpu = args.gpu if args.gpu else input('select gpus (e.g. 0 or 0,1 or 0,1,3,4): ').lower()
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

    scripts_path = os.path.join(config.experiment.exp_dir, config.experiment.exp_name, 'scripts')

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

    op_run = 'y' if args.y else input('run scripts ? (y/n): ').lower()
    if op_run in ['y', 'yes']:
        print('start running scripts...')
    else:
        exit(0)

    print('===== running =====')
    if not args.foreground:
        for f in all_script_files:
            subprocess.run('nohup sh {} > {} &'.format(f, args.to), shell=True)
    else:
        proc = [subprocess.Popen('sh {} > {}'.format(f, args.to), shell=True) for f in all_script_files]
        exit_codes = [p.wait() for p in proc]
        print('finished:', exit_codes)


def _expand_config(config):
    flattened_config = _flatten_config(config)
    keys = [item[0] for item in flattened_config]
    values = [item[1] if isinstance(item[1], list) else [item[1]] for item in flattened_config]
    configs = []
    for single_values in product(*values):
        assert len(keys) == len(single_values)
        data = list(zip(keys, single_values))
        configs.append(_create_config(data))
    return configs


def _flatten_config(obj, path=[]):
    if isinstance(obj, (str, int, float)):
        return [(path, obj)]
    elif isinstance(obj, dict):
        ret = []
        for k, v in obj.items():
            ret.extend(_flatten_config(v, path + [k]))
        return ret
    elif isinstance(obj, list):
        if all(isinstance(x, (str, int, float)) for x in obj):
            return [(path, obj)]
        else:
            ret = []
            for i, x in enumerate(obj):
                ret.extend(_flatten_config(x, path + [i]))
            return ret
    else:
        raise ValueError('Unsupported value type: {}'.format(type(obj)),
                         'Only the following value types are supported: str, int, float')


def _create_config(data):
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
    """ Collect experiment results launched by tune and save into a csv file """
    from modelscope.utils.config import Config
    assert args.cfg_file is not None, 'cfg_file must be specified!'
    config = Config.from_file(args.cfg_file)
    assert 'experiment' in config
    assert 'exp_name' in config.experiment

    output_path = os.path.join(config.experiment.exp_dir, config.experiment.exp_name, 'outputs')

    keys = None
    records = []
    log_files = glob.glob(os.path.join(output_path, '*/*.log.json'))
    for log_file in tqdm(log_files):
        try:
            result = _parse_log(log_file)
        except Exception:
            print('error during parsing log file {}'.format(log_file))
            continue
        records.append(list(result.values()) + [log_file])
        keys = list(result.keys()) + ['log_file']

    df = pd.DataFrame.from_records(records, columns=keys)
    df.to_csv(args.output_file)

    df_seed_avg = df.groupby(
        by=[k for k in list(df.columns)
            if k not in ['experiment_seed', 'p', 'r', 'f1', 'dev_f1', 'log_file']]).agg(['mean', 'std'])
    df_seed_avg.to_csv(args.output_avg_file)


IGNORE_KEY_REGEX = re.compile('dataset|evaluation|experiment_exp|hooks')


def _parse_log(log_file):
    ret = OrderedDict(dev_f1='NaN', p='NaN', r='NaN', f1='NaN')
    best_dev = -1
    with open(log_file, 'r') as f:
        hp = json.loads(f.readline())
        hp_list = _flatten_config(hp)
        for k, v in hp_list:
            k = '_'.join(map(str, k))
            if IGNORE_KEY_REGEX.search(k):
                continue
            ret[k] = v
        for line in f:
            if '"eval"' in line:
                result = json.loads(line)
                if result['f1'] > best_dev:
                    best_dev = result['f1']
                ret['dev_f1'] = best_dev
            elif '"test"' in line:
                result = json.loads(line)
                ret['p'] = result['precision']
                ret['r'] = result['recall']
                ret['f1'] = result['f1']
    return ret


def kill():
    """ Kill all running processes launched by tune """
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
    parser.add_argument('mode', choices=['tune', 'collect', 'kill'], help='[tune, collect, kill]')
    parser.add_argument('-c', '--cfg_file', help='configuration YAML file')
    parser.add_argument(
        '-c_env',
        '--config_env',
        help='configuration YAML file for environment',
        default='examples/grid_search/env.yaml')
    parser.add_argument('-to', '--to', help='stdout and stderr to', default='/dev/null')
    parser.add_argument('-o', '--output_file', help='output file for collect', default='res.csv')
    parser.add_argument('-oa', '--output_avg_file', help='output avg file for collect', default='res_seed_avg.csv')
    parser.add_argument('-g', '--gpu', help='gpu', default='')
    parser.add_argument('-y', '--y', help='default yes', action='store_true')
    parser.add_argument('-f', '--foreground', help='run in foreground and wait for exits for dlc', action='store_true')

    args = parser.parse_args()

    if args.mode == 'tune':
        tune(args)
    elif args.mode == 'collect':
        collect(args)
    else:
        kill()
