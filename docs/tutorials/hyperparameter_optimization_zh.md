# 超参数优化

这部分的教程介绍了一个用于快速调参的 grid search 工具。

grid search 工具包含3种功能：

- `tune`: 调参，以多组超参数配置并行跑实验
- `collect`: 收集，跑完实验后收集和分析实验结果
- `kill`: 停止，杀死所有的正在运行中的实验

#### 内容索引：
- [修改配置文件](#修改配置文件)
  - [[可选] 环境配置文件](#可选-环境配置文件)
- [用法](#用法)
  - [tune](#tune)
  - [collect](#collect)
  - [kill](#kill)

## 修改配置文件
在一切开始前，需要对配置文件进行一些修改。修改后的配置文件不能直接用于训练，仅用于生成实际运行使用的配置文件。

将需要调参的对象参数，从`k: v`的键值对，修改为列表形式：
```
k: [v1, v2]
```
或
```
k:
  - v1
  - v2
```
比如修改为含有2个值`k: [v1, v2]`的列表，这样实际调参中会运行`k: v1`和`k: v2`两组实验。

需要同时对多个参数进行调参时，将多个参数同时设置为列表，grid search 工具会自动探索不同参数组合，最终进行笛卡尔积数量的实验。
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
以上面的配置片段为例，我们设置了`lr`为`[5.0e-5, 2.0e-5]`，`crf_lr`为`[5.0e-1, 5.0e-2]`，最后我们会得到 2x2 = 4 组实验用于调参。

### [可选] 环境配置文件
环境配置文件主要用于设置环境变量，如：
```yaml
python_interpreter: 'python'  # default
python_file: 'scripts.train'  # default
envs:
  - 'export TRANSFORMERS_CACHE=xxx'
  - 'export HF_DATASETS_CACHE=xxx'
```

## 用法
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
`tune` 将会解析元配置文件并自动生成不同参数组合用于调参实验，当前支持两种运行模式：

*[交互式]* 以交互的方式选择gpu和实验来运行
```shell
python scripts/grid_search.py tune -c ${cfg_file} [-to log.txt]
```

*[一键式]* 设置好一切后再运行
```shell
python scripts/grid_search.py tune -c ${cfg_file} -y -g 0,1,2,3 [-to log.txt]
```

### collect
当所有实验运行完毕，使用`collect`来收集实验结果。同时，也会提供同一组参数但seed不通的多组实验归并后的结果。
```shell
python scripts/grid_search.py collect -c ${cfg_file} -o results.csv -oa seed_avg_results.csv
```

### kill
`kill` 命令可用于停止实验进程。
```shell
python scripts/grid_search.py kill
```
