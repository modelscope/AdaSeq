# Training a Model & Configuration Explanation
This part of tutorial shows how you can train you own models using state-of-the-art methods.

## Training a model
We currently provide a preset script to train a model with a configuration file. You can also write your own code to assemble a training procedure following the [doc](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train).

### Using preset script
```shell
python scripts/train.py -c ${cfg_file} [-t ${trainer} --seed ${seed}]
```
- **cfg_file** (`str`): Path to the configuration file.
- **trainer** (`str`, optional, default: `None`): A trainer name defined in [metainfo](./uner/metainfo.py). If this argument is not set, the trainer name should be found in `trainer: ${trainer}` in the configuration file.
- **seed** (`int`, optional, default: `None`): The random seed used for everything. If this argument is not set, we will try to find it in `experiments.seed: ${seed}` in the configuration file. If neither is set, the seed will be set to `0`. If `seed = -1`， a random seed will be used.

## Resuming training
```shell
python scripts/train.py -c ${cfg_file} -cp ${checkpoint_path} [-t ${trainer} --seed ${seed}]
```
- **checkpoint_path** (`str`): Path to the checkpoint file.

## Configuration Explanation
Let's take [resume.yaml](./examples/bert_crf/configs/resume.yaml) as an example.

#### Experiment
```yaml
experiment:
  exp_dir: experiments/  # experiment root directory
  exp_name: resume  # experiment name
  seed: 42  # random seed
```

#### Task
```yaml
task: named-entity-recognition  # task name, used to configure task-specific dataset-builders
```

#### Dataset
4 types of dataset loading methods are supported:

1. Load dataset from ModelScope
```yaml
dataset:
  name_or_path: damo/resume
```

2. Load local dataset python script or folder with python script inside
```yaml
dataset:
  name_or_path: ${abs_path_to_py_script_or_folder}
```

3. Load local/remote dataset zip
```yaml
dataset:
  data_dir: ${path_to_dataset_zip}
  data_type: sequence_labeling
  data_format: column
```

4. Load local/remote dataset files
```yaml
dataset:
  data_files:
    train: ${path_to_train_file}
    valid: ${path_to_validation_file}
    test: ${path_to_test_file}
  data_type: sequence_labeling
  data_format: column
```

#### Preprocessor
```yaml
preprocessor:
  type: sequence-labeling-preprocessor
  model_dir: sijunhe/nezha-cn-base
  max_length: 150
  data_collator: SequenceLabelingDataCollatorWithPadding
```

#### Model
```yaml
model:
  type: sequence-labeling-model
  encoder:
    model_name_or_path: sijunhe/nezha-cn-base
  word_dropout: 0.1
  use_crf: true
```

#### Trainer
```yaml
trainer: ner-trainer
```

#### Training arguments
```yaml
train:
  max_epochs: 20
  dataloader:
    batch_size_per_gpu: 16
    workers_per_gpu: 1
  optimizer:
    type: AdamW
    lr: 5.0e-5
    crf_lr: 5.0e-1
  lr_scheduler:
    type: LinearLR
    start_factor: 1.0
    end_factor: 0.0
    total_iters: 20
```

#### Evaluation arguments
```yaml
evaluation:
  dataloader:
    batch_size_per_gpu: 128
    workers_per_gpu: 1
    shuffle: false
  metrics:
    - type: ner-metric
    - type: ner-dumper
      model_type: sequence_labeling
      dump_format: column
```
