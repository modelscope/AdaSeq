# 快速开始

本教程介绍如何安装 `AdaSeq` 并且使用它训练一个模型。

- [1. 需求环境和安装方式](#1-需求环境和安装方式)
    - [a. 直接使用源代码（目前方式）](#1a-直接使用源代码)
    - [b. 使用 pip 安装（未来可用）](#1b-使用-pip-安装未来可用)
- [2. 使用示例](#2-使用示例)
    - [a. 使用代码脚本（目前方式）](#2a-使用代码脚本)
    - [b. 使用命令行（未来可用）](#2b-使用命令未来可用)

## 1. 需求环境和安装方式

AdaSeq 基于 `Python version >= 3.7` 和 `PyTorch version >= 1.8`

### 1.a 直接使用源代码

```commandline
git clone https://github.com/modelscope/adaseq.git
cd adaseq
pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

### 1.b 使用 pip 安装（未来可用）
我们暂时还没有发布到pypi，未来可以使用这种方式。
```commandline
pip install adaseq
```

## 2. 使用示例

本节将以在 `resume` 数据集上训练 BERT-CRF 模型作为示例。
训练一个模型，你只需要编写一个配置文件，然后允许一个命令即可。

我们已经准备了一个 [resume.yaml](../../examples/bert_crf/configs/resume.yaml) 配置文件，来试试吧！

### 2.a 使用代码脚本

#### 2.a.1 训练模型
```
python scripts/train.py -c examples/bert_crf/configs/resume.yaml
```

#### (b) 测试模型
```
python scripts/test.py -w ${checkpoint_dir}
```

### 2.b 使用命令（未来可用）

#### 2.b.1 训练模型
```
adaseq train -c examples/bert_crf/configs/resume.yaml
```

#### 2.b.2 测试模型
```
adaseq test -w ${checkpoint_dir}
```
