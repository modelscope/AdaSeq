# Learning about Configs
AdaSeq uses configuration file to control model assembling, training and evaluation. The configuration file supports `yaml` `json` `jsonline` format.

## 1. Configurate File Organization
Let's take [resume.yaml](../../examples/bert_crf/configs/resume.yaml) as an example. A configuration file usually consists of the following fields:

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

## 2. Introduction to Global Parameters
Notice: Default = `/` means this parameter is compulsory.

#### 2.1 experiment
| Parameter |                                      Description                                      | Type |   Default   |
|:---------:|:-------------------------------------------------------------------------------------:|:----:|:-----------:|
|  exp_dir  |                                 experiment directory                                  | str  | experiments |
| exp_name  | experiment name. all outputs will be saved to `./${exp_dir}/${exp_name}/${datetime}/` | str  |   unknown   |
|   seed    |                                      random seed                                      | int  |     42      |

#### 2.2 task
`task` supports the following values (see [metainfo](../../adaseq/metainfo.py)):
- chinese-word-segmentation
- part-of-speech
- named-entity-recognition
- relation-extraction
- entity-typing

#### 2.3 dataset
Please refer to [Customizing Dataset](./customizing_dataset_zh.md) as the combination of dataset parameters is complex.

|  Parameter   |                                                               Description                                                               |     Type      |  Default   |
|:------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:----------:|
|     task     |                                                           task of the dataset                                                           |      str      |    None    |
|     name     |                                         modelscope dataset name, for example `damo/resume_ner`                                          |      str      |    None    |
|     path     |                                            huggingface dataset name, for example `conll2003`                                            |      str      |    None    |
|  data_file   |          data files, it can be an url, local directory or archive, it can be a dict containing `train` `valid` `test` as well           |   str/dict    |    None    |
|  data_type   |                                                   used to specify data loading method                                                   |      str      |    None    |
|  transform   |                                    dataset post processing, usually containing `name` `key` `scheme`                                    |     dict      |    None    |
|    labels    | label set, it can be a list `labels: ['O', 'B-ORG', ...]`, file or url`labels: PATH_OR_URL`, or a function counting labels from dataset | str/list/dict |    None    |
| access_token |                                       used to access private repos from modelscope or huggingface                                       |      str      |    None    |

#### 2.4 preprocessor
|    Parameter     |               Description                | Type | Default |
|:----------------:|:----------------------------------------:|:----:|:-------:|
|       type       |            preprocessor type             | str  |    /    |
|    model_dir     |       tokenizer name or directory        | str  |    /    |
|   is_word2vec    |       whether to use Lookup Table        | bool |  False  |
| tokenizer_kwargs |      other parameters for tokenizer      | dict |  None   |
|    max_length    | maximum sentence length (subtoken-level) | int  |   512   |

#### 2.5 data_collator
`data_collator` supports the following values (see [metainfo](../../adaseq/metainfo.py)):
- DataCollatorWithPadding
- SequenceLabelingDataCollatorWithPadding
- SpanExtractionDataCollatorWithPadding
- MultiLabelSpanTypingDataCollatorWithPadding
- MultiLabelConcatTypingDataCollatorWithPadding

#### 2.6 model
| Parameter |  Child-Parameter   |                                   Description                                   | Type | Default |
|:---------:|:------------------:|:-------------------------------------------------------------------------------:|:----:|:-------:|
|   type    |                    |                                   model type                                    | str  |    /    |
| embedder  |                    |         used to embed input ids to vectors, usually a pretrained model          | dict |  None   |
|     └     |        type        |       embedder type, optional when using modelscope or huggingface model        | str  |  None   |
|     └     | model_name_or_path | pretrained model name or path, supporting both modelscope or huggingface models | str  |    /    |
|  encoder  |                    |                   encode the sentence vector, such as `LSTM`                    | dict |  None   |
|     └     |        type        |                                  encoder type                                   | str  |    /    |
|  decoder  |                    |                        not available, under construction                        | dict |  None   |

#### 2.7 train
|  Parameter   |  Child-Parameter   |                                                                    Description                                                                     | Type  | Default |
|:------------:|:------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|
|   trainer    |                    |                                                                    trainer type                                                                    |  str  |  None   |
|  max_epochs  |                    |                                                        maximum number of epochs in training                                                        |  int  |    /    |
|  dataloader  |                    |                                                                 used to load data                                                                  | dict  |    /    |
|      └       | batch_size_per_gpu |                                                                 batch size per gpu                                                                 |  int  |    /    |
|      └       |  workers_per_gpu   |                                                            data loading workers per gpu                                                            |  int  |    0    |
|  optimizer   |                    |                                                                     optimizer                                                                      | dict  |  None   |
|      └       |        type        |                                                                   optimizer type                                                                   |  str  |    /    |
|      └       |         lr         |                                           learning rate for all parameters except specific param_groups                                            | float |    /    |
|      └       |      options       |                                         options used in optimizer, for example `grad_clip: max_norm: 2.0`                                          | dict  |  None   |
|      └       |    param_groups    |                                                   param_groups can have different learning rates                                                   | list  |  None   |
|      └       |      └ regex       |                                                    regex expression to specify parameter group                                                     |  str  |    /    |
|      └       |        └ lr        |                                                     learning rate for specific parameter group                                                     | float |    /    |
| lr_scheduler |                    |                                                    used to adjust learning rate uding training                                                     | dict  |  None   |
|      └       |        type        |                               supporting all lr_scheduler from pytorch (check if your pytorch version includes them)                               |  str  |    /    |
|      └       |      options       |                                                          options used in the lr_scheduler                                                          | dict  |  None   |
|    hooks     |                    | also callbacks see [ModelScope documentation](https://modelscope.cn/docs/%E5%9B%9E%E8%B0%83%E5%87%BD%E6%95%B0%E6%9C%BA%E5%88%B6%E8%AF%A6%E8%A7%A3) | list  |  None   |

#### 2.8 evaluation
| Parameter  |  Child-Parameter   |         Description          | Type | Default |
|:----------:|:------------------:|:----------------------------:|:----:|:-------:|
| dataloader |                    |      used to load data       | dict |    /    |
|     └      | batch_size_per_gpu |      batch size per gpu      | int  |    /    |
|     └      |  workers_per_gpu   | data loading workers per gpu | int  |    0    |
|  metrics   |                    |      evaluation metrics      | list |  None   |
|     └      |        type        |         metric type          | str  |    /    |
