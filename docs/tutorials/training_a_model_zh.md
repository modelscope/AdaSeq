# 在自定义数据上训练模型

通常来说，`Model = Data + Network + Training`.

`AdaSeq` 目前支持文件形式的配置文件，您可以在训练之前写入各种参数。
一般情况下只需要一个配置文件即可训练一个模型。

本教程将分步介绍如何在自己的数据集上撰写配置文件，并训练模型。

<!-- TOC -->
- [1. 编写配置文件](#1-编写配置文件)
  - [1.1 元设置](#11-元设置)
  - [1.2 准备数据集](#12-准备数据集)
  - [1.3 设置模型结构相关](#13-设置模型相关结构)
  - [1.4 设置训练参数](#14-指定训练参数)
- [2. 开始训练](#2-开始训练)
  - [2.1 训练技巧](#21-训练技巧)
- [3. 获取评价结果和预测输出](#3-获取评价结果和预测输出)
- [4. 训练完成后](#4-训练完成后)
  - [4.1 模型推理](#41-模型推理)
  - [4.2 模型共享和在线调用](#42-模型共享和在线调用)
<!-- TOC -->

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
  data_file:  # 数据文件
    train: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=train.txt'
    valid: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=dev.txt'
    test: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=test.txt'
  data_type: conll  # 数据格式
```
一个示例配置如上，`AdaSeq` 将会尝试从远程链接获取训练 (train)、验证 (valid/dev) 和测试 (test) 集。
数据格式被指定为 `conll`，`AdaSeq` 将会自动使用相应的内建脚本加载数据集。

> 更多的数据集加载方法和数据集自定义方式，请查阅 [自定义数据集](./customizing_dataset_zh.md)。


### 1.3 设置模型相关结构

这里将设置定义模型计算流程的 `task` `preprocessor` `data_collator` `model`。

基本的数据流程是：
`dataset -> preprocessor -> data_collator -> model`

`preprocessor` 决定 AdaSeq 如何处理一个数据样例。
它需要一个 `model_dir` 参数来指定使用什么 `tokenizer` (`transformers`-style) 来将文本转换为数字id。如果不指定 `model_dir`，AdaSeq将会尝试使用embedder模型对应的 `tokenizer`。

`data_collator` 决定如何把多个处理好的样本聚合成一个批 (batch)。

`model` 设置具体的模型信息，`type` 指定使用哪个模型。
一个模型通常拥有 `embeder`, `encoder` 等可自定义替换的组件，更多参数清查看相关模型的代码定义，文档正在路上。

```yaml
task: named-entity-recognition  # 任务名称，用于加载内建的 DatasetBuilder（如果需要的话）

preprocessor:
  type: sequence-labeling-preprocessor  # 预处理器名称
  model_dir: bert-base-chinese  # huggingface/modelscope 模型名字或路径，用于初始化 Tokenizer，可缺省
  max_length: 150  # 预训练模型支持的最大输入长度

data_collator: SequenceLabelingDataCollatorWithPadding  # 用于 batch 转换的 data_collator 名称

model:
  type: sequence-labeling-model  # 模型名称
  embedder:
    model_name_or_path: damo/nlp_raner_named-entity-recognition_chinese-base-news  # 预训练模型名称或路径，可以是huggingface/modelscope的backbone模型，或者也可以加载modelscope上的任务模型
  dropout: 0.1   # dropout 概率
  use_crf: true  # 是否使用CRF
```

### 1.4 指定训练参数

最后，需要在配置文件的 `train` 和 `evaluation` 中设置训练和验证参数。这些参数很大程度上决定了所训练的模型的性能。

这一部分十分易于理解，这是所有深度学习框架通用的概念，你可以直接复制一份然后调整。

```yaml
train:
  max_epochs: 20  # 最大训练轮数
  dataloader:
    batch_size_per_gpu: 16  # 训练batch_size
  optimizer:
    type: AdamW  # pytorch 优化器名称
    lr: 5.0e-5  # 全局学习率
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
    batch_size_per_gpu: 128  # 评估batch_size
  metrics:
    - type: ner-metric  # 所有已实现的metric见 `adaseq/metainfo.py` 的 `Metrics` 类。
    - type: ner-dumper  # 输出预测结果
      model_type: sequence_labeling
      dump_format: column
```

## 2. 开始训练

当你准备好一个配置文件，训练模型很简单。你也可以使用 `examples` 文件夹中现成的配置文件来试试。

```commandline
adaseq train -c examples/bert_crf/configs/resume.yaml
```

> 如果你是通过git clone方式下载的本项目，请使用`python scripts/train.py -c examples/bert_crf/configs/resume.yaml`来运行。

更多的命令参数请查看 `scripts/train.py` 的注释 [TODO]。

### 2.1 训练技巧

- 实验记录可视化

  AdaSeq支持使用Tensorboard对训练过程中的学习率、loss、评估结果等指标进行可视化，可以辅助判断训练是否收敛、结果是否符合预期、比较不同参数组合的实验指标。参考如下配置：
  ```
  train:
    hooks:
      - type: TensorboardHook
  ```

- 替换backbone (embedder)

  AdaSeq支持使用Huggingface或ModelScope的所有Encoder-Only结构模型作为backbone，也支持加载已经训练好的任务模型中的backbone，如：
  - bert-base-cased
  - xlm-roberta-large
  - damo/nlp_structbert_backbone_base_std
  - damo/nlp_raner_named-entity-recognition_chinese-base-news

  参考如下配置：
  ```
  model:
    embedder:
      model_name_or_path: ${model_id}
  ```

- batch_size与梯度累积

  大的batch_size有助于提升模型的泛化能力。参考如下配置：
  ```
  train:
    dataloader:
      batch_size_per_gpu: 32
  ```

  当GPU显存不足以支持较大的batch_size时，建议通过梯度累积来模拟大的batch_size。比如让训练器每2个batch才进行一次反向传播，可以参考如下配置：
  ```
  train:
    dataloader:
      batch_size_per_gpu: 16
    optimizer:
      options:
        cumulative_iters: 2
  ```

- 分层学习率

  实验表明，在使用预训练模型作为backbone时，分层学习率可以有效提高学习效率和模型性能。AdaSeq支持通过正则表达式的方式配置参数组，以设置独立的学习率。参考如下配置：
  ```
  train:
    optimizer:
      lr: 5.0e-5  # 全局学习率
      param_group:
        - regex: crf  # 正则表达式
          lr: 5.0e-1  # 参数组的学习率
  ```

- 学习率调度器

  相比于恒定的学习率，使用自适应的学习率可以缩短训练时间、提高模型性能。AdaSeq目前支持torch所有的lr_scheduler，用户只需要在配置文件中配置lr_scheduler的类型和参数即可。

  一个简单的方法是让学习率随着时间的推移而不断衰减，可以参考如下配置：
  ```
  train:
    lr_scheduler:
      type: LinearLR
      start_factor: 1.0
      end_factor: 0.0
      total_iters: 20
  ```

- 周期性保存checkpoint

  AdaSeq[默认配置](./../../adaseq/training/default_config.py)下，仅会保存最优的checkpoint。如需按周期保存checkpoint，可以参考如下配置：
  ```
  train:
    hooks:
      - type: CheckpointHook
        interval: 1
        by_epoch: true
  ```

- [TODO] 继续训练

> 如果您想调整更多训练设置，请查阅以下文档。
> - [超参数优化](./hyperparameter_optimization_zh.md)
> - [训练加速](./training_acceleration_zh.md)

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

## 4. 训练完成后
### 4.1 模型推理
> 参考 [模型推理](./model_inference_zh.md)

### 4.2 模型共享和在线调用
> 参考 [模型发布到 ModelScope](./uploading_to_modelscope_zh.md)
