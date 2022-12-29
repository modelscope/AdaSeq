# Customizing Dataset

This part of tutorial shows how to prepare a custom dataset.

- [1. Overview](#1-overview)
- [2. Loading Dataset](#2-loading-dataset)
    - [from ModelScope](#21-loading-dataset-from-modelscope)
    - [from Huggingface](#22-loading-dataset-from-huggingface)
    - [via dataset loading script](#23-loading-dataset-via-dataset-loading-script)
    - [dataset files](#24-loading-dataset-files)
    - [dataset directory or archive](#25-loading-dataset-directory-or-archive)
- [3. Supported Dataset Formats](#3-supported-dataset-formats)
    - [3.1 Sequence Labeling Tasks](#31-sequence-labeling-tasks)
        - [CoNLL format](#311-conll-format)
        - [json tags format](#312-json-tags-format)
        - [json spans format](#313-json-spans-format)
        - [CLUENER json format](#314-cluener-format)

## 1. Overview

If you want to load a custom dataset, you should figure out 2 problems:

1. Where is the dataset?
2. Is the data format already supported?

In the following sections, we will introduce various dataset loading alternatives and different data formats we
support (more and more on the way).

## 2. Loading Dataset

Currently, 5 types of dataset loading methods are supported. You can set it in the configuration file.

#### 2.1 Loading dataset from ModelScope

```yaml
dataset:
  name: ${modelscope_dataset_name}
  access_token: ${access_token}
```

> `name` should be one of the uploaded datasets in ModelScope, such
> as [damo/resume_ner](https://modelscope.cn/datasets/damo/resume_ner/summary). `access_token` is NOT necessary unless the
> dataset is private.

#### 2.2 Loading dataset from Huggingface

```yaml
dataset:
  path: ${huggingface_dataset_name}
```

> `path` should be one of the uploaded datasets in Huggingface, such
> as [conll2003](https://huggingface.co/datasets/conll2003).

#### 2.3 Loading dataset via dataset loading script

```yaml
dataset:
  path: ${path_to_py_script_or_folder}
```

> `path` should be the absolute path to a custom [python script](https://huggingface.co/docs/datasets/dataset_script)
> for the `datasets.load_dataset` or a directory containing the script.

#### 2.4 Loading dataset files

```yaml
dataset:
  data_file:
    train: ${path_to_train_file}
    valid: ${path_to_validation_file}
    test: ${path_to_test_file}
  data_type: ${data_format}
```

> `train` `valid` `test` could be the urls or local paths (absolute paths) to the dataset files.
> `data_type` should be one of the supported data formats such as `conll`.

#### 2.5 Loading dataset directory or archive

```yaml
dataset:
  data_file: ${path_or_url_to_dir_or_archive}
  data_type: ${data_format}
```

> `data_file` could be an url like `"https://data.deepai.org/conll2003.zip"`, or a local directory (absolute path)
> like `"/home/data/conll2003"`,
> or a local archive file (absolute path) like `"/home/data/conll2003.zip"`.
> Also `data_type` should be one of the supported data formats such as `conll`.

## 3. Supported Dataset Formats

### 3.1 Sequence Labeling Tasks

For example, NER, CWS, POS Tagging, etc.

#### 3.1.1 CoNLL format

The widely-used CoNLL format is a specific vertical format (like TSV) that represents a tagged dataset. Normally it is a
text file with one word per line with sentences separated by ***an empty line***. The first column in a line should be a
word and the last column should be the word's tag (usually from `BIO` or `BIOES`).

Data Example:

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

我  O
是  O
另  O
一  O
句  O
话  O
```

> To use CoNLL format, set `data_type: conll`. Optionally, you can use `delimiter: ${custom_delimiter}` to set a custom
> delimiter for the conll file. By default, the delimiter is whitespace or tab.

#### 3.1.2 json-tags format

The json-tags format is similar to CoNLL format, where each sentence contains a 'text' field and a 'labels' field. The
length of 'text' and 'labels' should be exactly equal to each other, so we can assign all labels to its corresponding
character.

```
{
    "text": "鲁迅文学院组织有关专家",
    "labels": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O", "O"]
}
```

> To use CoNLL format, set `data_type: json_tags`.

#### 3.1.3 json-spans format

The json-spans format is another widely used format for both flat NER and nested NER. Each meaningful span is
represented as a dict with `start` `end` `type` field, indicating the [start, end) offsets, and the type of the span.

```
{
    "text": "鲁迅文学院组织有关专家",
    "spans": [{"start": 0, "end": 5, "type": "ORG"}, ...]
}
```

What's more, we allow `type` to be a list of labels, which means multi-label tagging is possible.

```
{
    "text": "人民日报出版社新近出版了王梦奎的短文集《翠微居杂笔》。",
    "spans": [{"start": 0, "end": 7, "type": ["组织", "出版商", "出版社"]}, ...]
}
```

> To use CoNLL format, set `data_type: json_spans`.

#### 3.1.4 CLUENER format

The CLUENER format is the official format used in the CLUENER benchmark, which gathers entities of the same type in a
group.

```
{
    "text": "鲁迅文学院组织有关专家",
    "label": {'ORG': [[0, 5], ...]}
}
```

> To use CLUENER format, set `data_type: cluener`.
