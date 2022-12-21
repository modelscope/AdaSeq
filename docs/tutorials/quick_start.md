# Quick Start

This tutorial introduces how to install AdaSeq and use it train a model.

- [1. Requirements and Installation](#1-requirements-and-installation)
    - [a. Installation from source](#1a-installation-from-source)
    - [b. Installation via pip](#1b-installation-via-pip)
- [2. Example Usage](#2-example-usage)
    - [a. Usage by code](#2a-usage-by-code)
    - [b. Usage via command-line tool](#2b-usage-via-command-line-tool)

## 1. Requirements and Installation

AdaSeq project is based on `Python version >= 3.7` and `PyTorch version >= 1.8`.

### 1.a Installation from source

```commandline
git clone https://github.com/modelscope/adaseq.git
cd adaseq
pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

### 1.b Installation via pip
```commandline
pip install adaseq
```

## 2. Example Usage

Let's train a Bert-CRF model for NER on the `resume` dataset as an example. All you need to do is to write a
configuration file and use it to run a command.

We've already prepared a configuration file [resume.yaml](../../examples/bert_crf/configs/resume.yaml) for you. Try it!

### 2.a Usage by code

#### 2.a.1 Train a model
```
python scripts/train.py -c examples/bert_crf/configs/resume.yaml
```

#### (b) Test a model
```
python scripts/test.py -w ${checkpoint_dir}
```

### 2.b Usage via command-line tool

#### 2.b.1 Train a model
```
adaseq train -c examples/bert_crf/configs/resume.yaml
```

#### 2.b.2 Test a model
```
adaseq test -w ${checkpoint_dir}
```
