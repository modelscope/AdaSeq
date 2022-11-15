# Preparing Custom Dataset
This part of tutorial shows how to prepare a custom dataset.

## Dataset format
#### Sequence labeling CoNLL format
```
鲁  B-ORG
迅  I-ORG
文  I-ORG
学  I-ORG
院  I-ORG
组  O
织  O
有  O
关  O
专  O
家  O
```

#### Sequence labeling json format
```
{"text": "鲁迅文学院组织有关专家", "labels": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O", "O"]}
```

#### Span-based / Entity typing json format
```
{"text": "人民日报出版社新近出版了王梦奎的短文集《翠微居杂笔》。", "label": [{"start": 0, "end": 7, "type": ["组织", "出版商", "出版社"]}]}
```

## Loading dataset

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
