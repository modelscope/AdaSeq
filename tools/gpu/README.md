# queue.py
A simple script to parallely run shell commands on multiple gpus. Commands will wait util there are free gpus.

### Quick Start
```commandline
python tools/gpu/queue.py command_file
```

### Usage
```
usage: queue.py [-h] [-g GPU] [-p MAX_GPU_POWER] [-m MAX_GPU_MEMORY]
                [-t TIME_INTERVAL] [-to LOG_FILE]
                commands

positional arguments:
  commands              a file in which each line contains a shell command.

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     gpu_ids, e.g. -g 0,1, default: all
  -p MAX_GPU_POWER, --max_gpu_power MAX_GPU_POWER
                        maximum gpu power(W) before running command, default:
                        100
  -m MAX_GPU_MEMORY, --max_gpu_memory MAX_GPU_MEMORY
                        maximum gpu memory(M) before running command, default:
                        1000
  -t TIME_INTERVAL, --time_interval TIME_INTERVAL
                        sleep for seconds between commands, default: 300
  -to LOG_FILE, --log_file LOG_FILE
                        redirect stdout to this log file, default: queue.log
```

### Command File Example
```commandline
adaseq train -c examples/SemEval2023_MultiCoNER_II/configs/orig/bn.yaml
adaseq train -c examples/SemEval2023_MultiCoNER_II/configs/orig/en.yaml
adaseq train -c examples/SemEval2023_MultiCoNER_II/configs/wiki128/bn.yaml
adaseq train -c examples/SemEval2023_MultiCoNER_II/configs/wiki128/en.yaml
```
