# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os
import subprocess
import time
from datetime import datetime
from typing import List, Tuple


def gpu_num():
    return len(os.popen('nvidia-smi | grep %').read().strip().split('\n'))


def gpu_info(gpu_id: int = 0) -> Tuple[int, int]:
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_id].split('|')
    gpu_power = int(gpu_status[1].split()[-3][:-1])
    gpu_memory = int(gpu_status[2].split('/')[0].strip()[:-3])
    return gpu_power, gpu_memory


def parse_gpu(gpu_ids: str) -> List[int]:
    return list(set([int(g) for g in gpu_ids.split(',')]))


def queue(
    commands: List[str],
    gpu_ids: List[int],
    max_gpu_power: int = 100,
    max_gpu_memory: int = 1000,
    time_interval: int = 300,
    log_file: str = 'queue.log',
) -> None:
    print('\n' + '=' * 50)
    print('Start Queueing!')

    if os.path.isfile(log_file):
        os.remove(log_file)

    for command in commands:
        while True:
            use_gpu_id = -1
            for gpu_id in gpu_ids:
                gpu_power, gpu_memory = gpu_info(gpu_id)
                if gpu_power <= max_gpu_power and gpu_memory <= max_gpu_memory:
                    use_gpu_id = gpu_id
                    break
            if use_gpu_id >= 0:
                gpu_command = f'CUDA_VISIBLE_DEVICES={use_gpu_id} nohup {command} >> {log_file} &'
                print(datetime.now())
                print(gpu_command)
                subprocess.run(gpu_command, shell=True)
                time.sleep(time_interval)
                break
            time.sleep(time_interval)


def main(args):
    n_gpu = gpu_num()
    if args.gpu == 'all':
        gpu_ids = list(range(n_gpu))
    else:
        gpu_ids = parse_gpu(args.gpu)

    assert all(map(lambda g: g < n_gpu, gpu_ids)), 'Arg gpu_ids exceed total gpu number!'

    commands = []
    with open(args.commands, 'r') as f:
        for line in f:
            commands.append(line.strip())

    print('\n' + '=' * 50)
    print(f'Use GPU: {gpu_ids}')
    print('Commands to Run:')
    for command in commands:
        print(command)

    queue(
        commands,
        gpu_ids,
        args.max_gpu_power,
        args.max_gpu_memory,
        args.time_interval,
        args.log_file,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'commands', type=str, help='a file in which each line contains a shell command.'
    )
    parser.add_argument(
        '-g', '--gpu', type=str, default='all', help='gpu_ids, e.g. -g 0,1, default: all'
    )
    parser.add_argument(
        '-p',
        '--max_gpu_power',
        type=int,
        default=100,
        help='maximum gpu power(W) before running command, default: 100',
    )
    parser.add_argument(
        '-m',
        '--max_gpu_memory',
        type=int,
        default=1000,
        help='maximum gpu memory(M) before running command, default: 1000',
    )
    parser.add_argument(
        '-t',
        '--time_interval',
        type=int,
        default=300,
        help='sleep for seconds between commands, default: 300',
    )
    parser.add_argument(
        '-to',
        '--log_file',
        type=str,
        default='queue.log',
        help='redirect stdout to this log file, default: queue.log',
    )
    args = parser.parse_args()

    main(args)
