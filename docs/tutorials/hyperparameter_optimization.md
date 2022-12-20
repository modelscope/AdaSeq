# Hyperparameter Optimization
This part of tutorial introduces a grid search tool for efficient tuning. It's nothing special, but it's practical.

The grid search tool contains 3 modes:
- `tune`: run experiments with multiple hyperparameter settings in parallel
- `collect`: collect and analyze the experimental results
- `kill`: kill all running grid search processes

#### Table of contents:
- [Modify configuration file](#modify-configuration-file)
  - [[Optional] Environment configuration file](#optional-environment-configuration-file)
- [Usage](#usage)
  - [tune](#tune)
  - [collect](#collect)
  - [kill](#kill)

## Modify configuration file
Before everything starts, some modification should be done to the configuration file first.
Tuning a hyperparameter is easy, all you need to do is changing the hyperparameter to **LIST** in the configuration file.
```yaml
train:
  optimizer:
    type: AdamW
    lr:
      - 5.0e-5
      - 2.0e-5
    param_groups:
      - regex: crf
        lr:
          - 5.0e-1
          - 5.0e-2
```
As an example, we set `lr` to `[5.0e-5, 2.0e-5]`, `crf_lr` to `[5.0e-1, 5.0e-2]`, and we will get 2x2 = 4 experiments for tuning.

### [Optional] Environment configuration file
You can also write an environment configuration file to set environment variables.
```yaml
python_interpreter: 'python'  # default
python_file: 'scripts.train'  # default
envs:
  - 'export TRANSFORMERS_CACHE=xxx'
  - 'export HF_DATASETS_CACHE=xxx'
```

## Usage
```
usage: grid_search.py [-h] [-c CFG_FILE] [-c_env CONFIG_ENV] [-to TO]
                      [-o OUTPUT_FILE] [-oa OUTPUT_AVG_FILE] [-g GPU] [-y]
                      [-f]
                      {tune,collect,kill}

positional arguments:
  {tune,collect,kill}   [tune, collect, kill]

optional arguments:
  -h, --help            show this help message and exit
  -c CFG_FILE, --cfg_file CFG_FILE
                        configuration YAML file
  -c_env CONFIG_ENV, --config_env CONFIG_ENV
                        configuration YAML file for environment
  -to TO, --to TO       stdout and stderr to
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output file for collect
  -oa OUTPUT_AVG_FILE, --output_avg_file OUTPUT_AVG_FILE
                        output avg file for collect
  -g GPU, --gpu GPU     gpu_ids (e.g. 0 or 0,1 or 0,1,3,4)
  -y, --yes             automatically answer yes for all questions
  -f, --foreground      run in foreground and wait for exits for dlc
```

### tune
`tune` allows running experiments with multiple hyperparameter settings in parallel. It supports two modes:

*[Interactive mode]* You can interactively select gpus and experiments to run.
```shell
python scripts/grid_search.py tune -c ${cfg_file} [-to log.txt]
```

*[Hand-free mode]* Set everything and run, then you can leave to have a coffee.
```shell
python scripts/grid_search.py tune -c ${cfg_file} -y -g 0,1,2,3 [-to log.txt]
```

### collect
After all experiments are finished, you can use `collect` to collect all experimental results. The averaged results grouped by seed is provided as well.
```shell
python scripts/grid_search.py collect -c ${cfg_file} -o results.csv -oa seed_avg_results.csv
```

### kill
`kill` command is used to stop all tuning processes.
```shell
python scripts/grid_search.py kill
```
