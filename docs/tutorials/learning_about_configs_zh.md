# 了解配置文件
AdaSeq目前使用配置文件来控制模型的组装、训练和评估，配置文件支持`yaml` `json` `jsonline`格式。

## 1. 配置文件结构
以[resume.yaml](../../examples/bert_crf/configs/resume.yaml)为例，一个配置文件通常包括下面几个域：

```yaml
experiment: ...
task: ...
dataset: ...
preprocessor: ...
data_collator: ...
model: ...
train: ...
evaluation: ...
```

## 2. 全局参数介绍
注：默认值为`/`表示该参数为必填项。

#### 2.1 experiment
|    参数    |                          说明                           | 参数类型 |     默认值     |
|:--------:|:-----------------------------------------------------:|:----:|:-----------:|
| exp_dir  |                         实验目录                          | str  | experiments |
| exp_name | 实验名称，所有输出将会保存到`./${exp_dir}/${exp_name}/${datetime}/` | str  |   unknown   |
|   seed   |                         随机数种子                         | int  |     42      |

#### 2.2 task
`task`无子参数，目前支持下列值（也可见[metainfo](../../adaseq/metainfo.py)）：
- word-segmentation
- part-of-speech
- named-entity-recognition
- relation-extraction
- entity-typing

#### 2.3 dataset
数据集参数组合较复杂，建议参考[自定义数据集](./customizing_dataset_zh.md)。

|      参数      |                                            说明                                             |     参数类型      | 默认值  |
|:------------:|:-----------------------------------------------------------------------------------------:|:-------------:|:----:|
|     task     |                                           任务类型                                            |      str      | None |
|     name     |                            modelscope数据集名称，如`damo/resume_ner`                             |      str      | None |
|     path     |                               huggingface数据集名称，如`conll2003`                               |      str      | None |
|  data_file   |                 数据文件，可以是url、本地目录或本地压缩包，也可以是一个包含`train` `valid` `test`的字典                  |   str/dict    | None |
|  data_type   |                                      数据格式，用于指定数据读取方法                                      |      str      | None |
|  transform   |                             数据后处理，可包含`name` `key` `scheme`等字段                             |     dict      | None |
|    labels    | 标签集，可以直接传入标签列表`labels: ['O', 'B-ORG', ...]`，或传入标签文件或url`labels: PATH_OR_URL`，或设置函数从数据集中统计 | str/list/dict | None |
| access_token |                             用于访问modelscope或huggingface的私有数据仓库                             |      str      | None |

#### 2.4 preprocessor
|        参数        |           说明           | 参数类型 |  默认值  |
|:----------------:|:----------------------:|:----:|:-----:|
|       type       |     preprocessor类型     | str  |   /   |
|    model_dir     |     tokenizer名称或目录     | str  |   /   |
|   is_word2vec    |    是否使用Lookup Table    | bool | False |
| tokenizer_kwargs |     tokenizer其他参数      | dict | None  |
|    max_length    | 最大句子长度(subtoken-level) | int  |  512  |

#### 2.5 data_collator
`data_collator`无子参数，目前支持下列值（也可见[metainfo](../../adaseq/metainfo.py)）：
- DataCollatorWithPadding
- SequenceLabelingDataCollatorWithPadding
- SpanExtractionDataCollatorWithPadding
- MultiLabelSpanTypingDataCollatorWithPadding
- MultiLabelConcatTypingDataCollatorWithPadding

#### 2.6 model
|    参数    |        子参数         |            说明            | 参数类型  | 默认值  |
|:--------:|:------------------:|:------------------------:|:-----:|:----:|
|   type   |                    |           模型类型           |  str  |  /   |
| embedder |                    |     表征学习器，通常是一个预训练模型     | dict  | None |
|    └     |        type        |  embedder类型，使用ms/hf时可不填  |  str  | None |
|    └     | model_name_or_path |    预训练模型名称或路径，支持ms&hf    |  str  |  /   |
| encoder  |                    | 对句子表征做进一步encode，如`LSTM`等 | dict  | None |
|    └     |        type        |        encoder类型         |  str  |  /   |
| decoder  |                    |           开发中            | dict  | None |

#### 2.7 train
|      参数      |        子参数         |                                                              说明                                                              | 参数类型  | 默认值  |
|:------------:|:------------------:|:----------------------------------------------------------------------------------------------------------------------------:|:-----:|:----:|
|   trainer    |                    |                                                            训练器类型                                                             |  str  | None |
|  max_epochs  |                    |                                                           最大epoch数                                                           |  int  |  /   |
|  dataloader  |                    |                                                            数据读取器                                                             | dict  |  /   |
|      └       | batch_size_per_gpu |                                                      每块gpu上的batch size                                                       |  int  |  /   |
|      └       |  workers_per_gpu   |                                                        每块gpu上的数据读取进程数                                                        |  int  |  0   |
|  optimizer   |                    |                                                             优化器                                                              | dict  | None |
|      └       |        type        |                                                            优化器类型                                                             |  str  |  /   |
|      └       |         lr         |                                                             学习率                                                              | float |  /   |
|      └       |      options       |                                            可指定优化器其他参数，如`grad_clip: max_norm: 2.0`                                            | dict  | None |
|      └       |    param_groups    |                                                     模型参数组，支持正则表达式自定义学习率                                                      | list  | None |
|      └       |      └ regex       |                                                       正则表达式，用于指定模型参数组                                                        |  str  |  /   |
|      └       |        └ lr        |                                                         特定模型参数组的学习率                                                          | float |  /   |
| lr_scheduler |                    |                                                            学习率规划器                                                            | dict  | None |
|      └       |        type        |                                学习率规划器类型，支持pytorch所有lr_scheduler（注意pytorch版本是否包含该lr_scheduler）                                |  str  |  /   |
|      └       |      options       |                                                          学习率规划器其他参数                                                          | dict  | None |
|    hooks     |                    | 回调函数，详见[ModelScope官方文档](https://modelscope.cn/docs/%E5%9B%9E%E8%B0%83%E5%87%BD%E6%95%B0%E6%9C%BA%E5%88%B6%E8%AF%A6%E8%A7%A3) | list  | None |

#### 2.8 evaluation
|     参数     |        子参数         |        说明         | 参数类型 | 默认值  |
|:----------:|:------------------:|:-----------------:|:----:|:----:|
| dataloader |                    |       数据读取器       | dict |  /   |
|     └      | batch_size_per_gpu | 每块gpu上的batch size | int  |  /   |
|     └      |  workers_per_gpu   |  每块gpu上的数据读取进程数   | int  |  0   |
|  metrics   |                    |       评价指标        | list | None |
|     └      |        type        |      评价指标类型       | str  |  /   |
