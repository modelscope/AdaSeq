# Grid Search for tuning hyperparameters
The grid search script contains 3 modes: [tune, collect, kill]

## tune - run experiments with multiple hyperparameter settings in parallel
```shell
python scripts/grid_search.py tune -c ${cfg_file} [-c_env ${env_file} -to ${log_file}]
```

## collect - collect and analyze the experimental results
```shell
python scripts/grid_search.py tune -c ${cfg_file} [-o ${result_file} -oa ${seed_avg_result_file}]
```

## kill - kill all running grid search processes
```shell
python scripts/grid_search.py kill
```
