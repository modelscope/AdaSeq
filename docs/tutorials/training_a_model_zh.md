# 模型训练方法 & 配置文件解读

通常来说，`Model = Data + Network + Training`.

`AdaSeq` 目前支持文件形式的配置文件，您可以在训练之前写入各种参数。
一般情况下只需要一个配置文件即可训练一个模型。

本教程将分步介绍如何在自己的数据集上撰写配置文件，并训练模型。


- [1. 编写配置文件](#1-编写配置文件)
  - [1.1 元设置](#11-元设置)
  - [1.2 准备数据集](#12-准备数据集)
  - [1.3 设置模型结构相关](#13-设置模型相关结构)
  - [1.4 设置训练参数](#14-指定训练参数)
- [2. 开始训练](#2-开始训练)
- [3. 获取评价结果和预测输出](#3-获取评价结果和预测输出)


## 1. 编写配置文件
本节将在 `resume` 数据上实现 BERT-CRF 模型，以已有的配置文件 [resume.yaml](../../examples/bert_crf/configs/resume.yaml) 为例。

> 配置参数的详细解释清查阅 [学习配置文件](./learning_about_configs_zh.md).

### 1.1 元设置
第一步，需要设置训练期间各种输出文件的存储位置。

```yaml
experiment:
  exp_dir: experiments/  # 所有实验的根文件夹
  exp_name: resume  # 本次配置文件的实验名称
  seed: 42  # 随机种子
```

由此，训练日志、模型存档、预测输出和当前配置文件的备份将存放于 `./experiments/resume/${RUN_NAME}/`。

> `RUN_NAME` 是本次实验运行的名称，可以通过训练脚本或训练指定的参数 `-r/--run_name` 指定。若未指定，默认以UNIX时间戳作为名称。


### 1.2 准备数据集
`dataset` 参数决定了数据集的各种形式和来源。

```yaml
dataset:
  data_file:
    train: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=train.txt'
    valid: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=dev.txt'
    test: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=test.txt'
  data_type: conll
```
一个示例配置如上，`AdaSeq` 将会尝试从远程链接获取训练 (train)、验证 (valid/dev) 和测试 (test) 集。
数据格式被指定为 `conll`，`AdaSeq` 将会自动使用相应的内建脚本加载数据集。

> 更多的数据集加载方法和数据集自定义方式，请查阅 [自定义数据集](./customizing_dataset_zh.md)。


### 1.3 设置模型相关结构

这里将设置定义模型计算流程的 `task` `preprocessor` `data_collator` `model`。

基本的数据流程是：
`dataset -> preprocessor -> data_collator -> model`

`preprocessor` 决定 AdaSeq 如何处理一个数据样例。
它需要一个 `model_dir` 参数来指定使用什么 `tokenizer` (`transformers`-style) 来将文本转换为数字id。

`data_collator` 决定如何把多个处理好的样本聚合成一个批 (batch)。

`model` 设置具体的模型信息，`type` 指定使用哪个模型。
一个模型通常拥有 `embeder`, `encoder` 等可自定义替换的组件，更多参数清查看相关模型的代码定义，文档正在路上。

```yaml
task: named-entity-recognition  # 任务名称，用于加载内建的 DatasetBuilder（如果需要的话）

preprocessor:
  type: sequence-labeling-preprocessor  # 名称
  model_dir: sijunhe/nezha-cn-base  # huggingface/modelscop 模型名字或路径，用于初始化 Tokenizer
  max_length: 150  # 预训练模型支持的最大输入长度

data_collator: SequenceLabelingDataCollatorWithPadding  # 用于 batch 转换的 data_collator 名称

model:
  type: sequence-labeling-model  # 模型名称
  embedder:
    model_name_or_path: sijunhe/nezha-cn-base  # 预训练模型名称或路径
  dropout: 0.1   # dropout 概率
  use_crf: true  # 是否使用CRF
```

### 1.4 指定训练参数

最后，需要在配置文件的 `train` 和 `evaluation` 中设置训练和验证参数。这些参数很大程度上决定了所训练的模型的性能。

这一部分十分易于理解，这是所有深度学习框架通用的概念，你可以直接复制一份然后调整。

```yaml
train:
  max_epochs: 20
  dataloader:
    batch_size_per_gpu: 16
  optimizer:
    type: AdamW  # pytorch 优化器名称
    lr: 5.0e-5
    param_group:  # 不同优化参数组的设置
      - regex: crf  # 正则表达式
        lr: 5.0e-1  # 本组参数的学习率
  lr_scheduler:
    type: LinearLR  # transformers 或 pytorch 的 lr_scheduler 名称
    start_factor: 1.0
    end_factor: 0.0
    total_iters: 20

evaluation:
  dataloader:
    batch_size_per_gpu: 128
  metrics:
    - type: ner-metric  # 所有已实现的metric见 `adaseq/metainfo.py` 的 `Metrics` 类。
    - type: ner-dumper
      model_type: sequence_labeling
      dump_format: column
```

## 2. 开始训练

当你准备好一个配置文件，训练模型很简单。你也可以使用 `examples` 文件夹中现成的配置文件来试试。

```commandline
python scripts/train.py -c examples/bert_crf/configs/resume.yaml
```
更多的命令参数请查看 `scripts/train.py` 的注释。

> 如果您想调整更多训练设置，请查阅以下文档。
> - [超参数优化](./hyperparameter_optimization_zh.md)
> - [多 GPU 训练](./training_with_multiple_gpus_zh.md)

## 3. 获取评价结果和预测输出

在训练过程中，进度和评测结果会同时在命令行终端和日志文件中展示。


参见 [1.1 元设置](#11-元设置) 所述，所有输出文件将会保存在 `./experiments/resume/${RUN_NAME}/`。
训练完成后，在此文件夹中将会有 5 个不同的文件。
```
./experiments/resume/${RUN_NAME}/
├── best_model.pth
├── config.yaml
├── metrics.json
├── out.log
└── pred.txt
```

你可以到 `metrics.json` 中收集评测性能结果，或者查看 `out.log` 保存的所有训练日志。
`pred.txt` 将为保存在测试集的预测结果。你可以使用它来分析并改进你的模型，或者将其提交到某个比赛。
`best_model.pth` 可以用来进一步微调或者部署。
