# Training a Model with Custom Dataset
Generally speaking, `Model = Data + Network + Training`.

AdaSeq currently supports file-based mode, which means you need to clarify these arguments in a configuration file before starting training. And the configuration file is all you need to train a model.

This part of tutorial will introduce how to train a model with a custom dataset step by step in the sections to come.

- [1. Writing a Configuration File](#1-writing-a-configuration-file)
  - [1.1 Meta Settings](#11-meta-settings)
  - [1.2 Preparing Dataset](#12-preparing-dataset)
  - [1.3 Assembling Model Architecture](#13-assembling-model-architecture)
  - [1.4 Specifying Training Hyper-parameters](#14-specifying-training-arguments)
- [2. Starting Training](#2-starting-training)
- [3. Fetching Evaluation Results and Predictions](#3-fetching-evaluation-results-and-predictions)

## 1. Writing a Configuration File
Let's take [resume.yaml](../../examples/bert_crf/configs/resume.yaml) as an example.

> For detailed descriptions of the configuration arguments, please refer to this tutorial [Learning about Configs](./learning_about_configs.md).

### 1.1 Meta Settings
First, tell AdaSeq where to save all outputs during training.

```yaml
experiment:
  exp_dir: experiments/
  exp_name: resume
  seed: 42
```
So the training logs, model checkpoints, prediction results and a copy of the configuration files will all be saved to `./experiments/resume/${datetime}/`

### 1.2 Preparing Dataset
The `dataset` argument determines what the dataset look like (data format) and where to fetch the dataset (data source).

```yaml
dataset:
  data_file:
    train: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=train.txt'
    valid: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=dev.txt'
    test: 'https://www.modelscope.cn/api/v1/datasets/damo/resume_ner/repo/files?Revision=master&FilePath=test.txt'
  data_type: conll
```
As the example snippet shows, AdaSeq will fetch the training, validation and testing dataset files from remote urls. The data is specified as `conll` format so that a corresponding script will be used to parse the data.

> For more dataset loading approaches and supported data formats, please refer to this tutorial [Customizing Dataset](./customizing_dataset.md).

### 1.3 Assembling Model Architecture
This part specifies the `task` `preprocessor` `data_collator` `model` in the training.

The basic data flow is:
`dataset -> preprocessor -> data_collator -> model`

`preprocessor` tells how AdaSeq processes a data sample. It needs a `model_dir` indicating the tokenizer to use and is used to turn a sentence into ids and masks.

`data_collator` tells how to collate data samples into batches.

`model` tells how a model is assembled where `type` indicates the basic architecture. A model usually consists of several replaceable components such as `embeder`, `encoder`, etc.

```yaml
task: named-entity-recognition

preprocessor:
  type: sequence-labeling-preprocessor
  model_dir: sijunhe/nezha-cn-base
  max_length: 150

data_collator: SequenceLabelingDataCollatorWithPadding

model:
  type: sequence-labeling-model
  embedder:
    model_name_or_path: sijunhe/nezha-cn-base
  dropout: 0.1
  use_crf: true
```

### 1.4 Specifying Training Arguments
Last but not least, set some training and evaluation arguments in `train` and `evaluation`. Model performances can vary widely under different training settings.

This part is relatively easy to understand (I think). You can copy one and adjust the values to whatever you want.
```yaml
train:
  max_epochs: 20
  dataloader:
    batch_size_per_gpu: 16
  optimizer:
    type: AdamW
    lr: 5.0e-5
    param_groups:
      - regex: crf
        lr: 5.0e-1
  lr_scheduler:
    type: LinearLR
    start_factor: 1.0
    end_factor: 0.0
    total_iters: 20

evaluation:
  dataloader:
    batch_size_per_gpu: 128
  metrics:
    - type: ner-metric
    - type: ner-dumper
      model_type: sequence_labeling
      dump_format: conll
```

## 2. Starting Training
Once you have a configuration file, it is easy to training a model. You can also use any of the configuration files in the examples. Just try it!

```commandline
python scripts/train.py -c examples/bert_crf/configs/resume.yaml
```

> We also provide advanced tutorials if you want to improve your training.
> - [Hyperparameter Optimization](./hyperparameter_optimization.md)
> - [Training with Multiple GPUs](./training_with_multiple_gpus.md)

## 3. Fetching Evaluation Results and Predictions
During training, the process bar and the evaluation results will be logged to both the terminal and to a log file.

As we mentioned in [1.1 Meta Settings](#11-meta-settings), all outputs will be saved to `./experiments/resume/${datetime}/`. After training, there will be 5 files in the folder.
```
./experiments/resume/${datetime}/
├── best_model.pth
├── config.yaml
├── metrics.json
├── out.log
└── pred.txt
```

You can either collect the evaluation results in `metrics.json` or review all training logs in `out.log`. `pred.txt` will give predictions on the test dataset. You can analyze it to improve your model or submit it to some competition. `best_model.pth` can be used for further tuning or deployment.
