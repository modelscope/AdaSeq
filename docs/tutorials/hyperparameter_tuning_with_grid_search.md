# Hyperparameter Tuning with Grid Search
This part of tutorial introduces a grid search tool for efficient tuning. It's nothing special, but it's practical.

The grid search tool contains 3 modes:
- `tune`: Run experiments with multiple hyperparameter settings in parallel
- `collect`: Collect and analyze the experimental results
- `kill`: Kill all running grid search processes

Before everything starts, let's modify our configuration file first.

## Modify configuration file
Tuning a hyperparameter is easy, all you need to do is changing the hyperparameter to **LIST** in the configuration file.
```yaml
  optimizer:
    type: AdamW
    lr:
      - 5.0e-5
      - 2.0e-5
    crf_lr:
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

## tune
```shell
python scripts/grid_search.py tune -c ${cfg_file} [-c_env ${env_file} -g ${gpu_ids} -y -f -to ${log_file}]
```


## collect
```shell
python scripts/grid_search.py collect -c ${cfg_file} [-o ${result_file} -oa ${seed_avg_result_file}]
```

## kill
```shell
python scripts/grid_search.py kill
```
