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

## tune
```shell
python scripts/grid_search.py tune -c ${cfg_file} [-c_env ${env_file} -to ${log_file}]
```

## collect
```shell
python scripts/grid_search.py collect -c ${cfg_file} [-o ${result_file} -oa ${seed_avg_result_file}]
```

## kill
```shell
python scripts/grid_search.py kill
```
