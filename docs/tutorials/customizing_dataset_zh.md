# 自定义数据集

本部分的教程介绍如何准备和使用自定义数据集。

- [1. 概览](#1-概览)
- [2. 加载数据集](#2-加载数据集)
    - [数据集train、dev、test划分文件（推荐）](#21-从本地或远程加载数据文件推荐)
    - [数据集目录或者压缩包（推荐）](#22-从-本地文件夹-或者-本地远程的压缩文件-加载推荐)
    - [从 ModelScope 加载](#23-从-modelscope-加载数据集)
    - [从 Huggingface 加载](#24-从-huggingface-加载数据集)
    - [使用Huggingface支持的数据集加载脚本](#25-使用自定义的加载脚本)
- [3. 内建加载脚本支持的数据集格式](#3-内建加载脚本支持的数据集格式)
    - [3.1 序列标注任务](#31-序列标注任务)
        - [CoNLL 格式](#311-conll-格式)
        - [json tags 格式](#312-json-tags-格式)
        - [json spans 格式](#313-json-spans-格式)
        - [CLUENER 格式](#314-cluener-格式)
- [4. 数据集后处理](#4-数据集后处理)
    - [4.1 数据集格式转换](#41-数据集格式转换)
    - [4.2 标签集获取](#42-标签集获取)
        - [从数据集统计](#421-从数据集统计标签)


## 1. 概览

如果想要加载和使用一个自定义数据集，首先需要明确两个问题：

1. 数据集存放在哪里？
2. 这种数据格式是否已经被 `adaseq` 支持？

在接下来的部分，会介绍不同的数据集加载方式和已经支持的不同数据集格式（更多的方式正在开发）。


## 2. 加载数据集

目前 `adaseq` 支持 5 种不同的数据集加载方式，都可以在配置文件中指定。

推荐使用前两种方式，即使用内建加载脚本从本地或者远程加载数据。如果使用其他方式，请相应阅读 [4.1 数据集格式转换](#41-数据集格式转换) 部分，进行必要的设置。


#### 2.1 从本地或远程加载数据文件（推荐）

`adaseq` 可以支持使用内建的 `python` 脚本（`adaseq/data/dataset_builders/` 目录下列出了所有脚本）加载本地或者远程的数据文件，`adaseq.data.DatasetManager` 将通过配置文件中的 `task` 字段自动使用相应的脚本。

本节介绍分别指定不同数据集划分名称的加载方式。

```yaml
dataset:
  data_file:
    train: ${path_to_train_file}
    valid: ${path_to_validation_file}
    test: ${path_to_test_file}
  data_type: ${data_format}
```

> `train` `valid` `test` 是相应数据集划分文件的 `url` 或者 本地绝对路径。

> `data_type` 是目前 `adaseq` 内建支持的数据格式名称，比如 `conll`，所有格式见第 3 节。


#### 2.2 从 本地文件夹 或者 本地/远程的压缩文件 加载（推荐）

`adaseq` 可以支持使用内建的 `python` 脚本（`adaseq/data/dataset_builders/` 目录下列出了所有脚本）加载本地或者远程的数据文件，`adaseq.data.DatasetManager` 将通过配置文件中的 `task` 字段自动使用相应的脚本。

本节介绍指定本地文件夹、本地压缩包、远程压缩包的方式。


```yaml
dataset:
  data_file: ${path_or_url_to_dir_or_archive}
  data_type: ${data_format}
```

> `data_file` 可以是远程压缩包链接，比如 `"https://data.deepai.org/conll2003.zip"`；
> 或者本地压缩包绝对路径，比如 `"/home/data/conll2003.zip"`；
> 或者本地文件夹绝对路径，比如 `"/home/data/conll2003"`。

> `data_type` 是目前 `adaseq` 内建支持的数据格式名称，比如 `conll`，所有格式见第 3 节。

**注意**：如果使用压缩包，各数据文件必须放置于压缩包根目录，也就是说压缩包内不能含有文件夹。


#### 2.3 从 ModelScope 加载数据集

目前仅内建支持从 ModelScope 加载 `bio` 格式的 NER 数据，您可以自行添加转换函数(`transform`)来支持更多任务（见第 4.1 节），我们正努力添加新的格式。

```yaml
dataset:
  name: ${modelscope_dataset_name}
  transform: hf_ner_to_adaseq
  access_token: ${access_token}
```

> `name` 是在 ModelScope 上已有数据集的名字，比如
> [`damo/resume_ner`](https://modelscope.cn/datasets/damo/resume_ner/summary)。

> `transform` 是可用的数据集格式转换函数名称，如果您的数据与 `adaseq` 内建格式不一致时，则需要指定此项，更多细节见第 4.1 节。

> `access_token` 是 ModelScope 的访问密钥，加载公开数据集 **不需要** 填写，只有当使用私有数据集时才需要。

#### 2.4 从 Huggingface 加载数据集

目前仅内建支持从 Huggingface 加载 `bio` 格式的 NER 数据，您可以自行添加转换函数(`transform`)来支持更多任务（见第 4.1 节），我们正努力添加新的格式。

```yaml
dataset:
  path: ${huggingface_dataset_name}
  transform: hf_ner_to_adaseq
```

> `path` 是 Huggingface 上数据集的名称, 比如
> [`conll2003`](https://huggingface.co/datasets/conll2003).

> `transform` 是可用的数据集格式转换函数名称，如果您的数据与 `adaseq` 内建格式不一致时，则需要指定此项，更多细节见第 4.1 节。


#### 2.5 使用自定义的加载脚本

要使用一个自定义的 `huggingface.datasets` 支持的数据加载 `python` 脚本，
您需要在此 `python` 脚本中将数据处理成 `adaseq` 支持的格式，不同任务的格式支持请查阅 `adaseq/data/dataset_builders/` 目录下的不同任务的内建加载脚本。

```yaml
dataset:
  path: ${path_to_py_script_or_folder}
```

> `path` 是自定义 [python script](https://huggingface.co/docs/datasets/dataset_script) 的绝对路径，或者包含此脚本的目录，
> 将被用于 `datasets.load_dataset` 函数，更多细节参见 `datasets` [相关文档](https://huggingface.co/docs/datasets/dataset_script)。



## 3. 内建加载脚本支持的数据集格式

### 3.1 序列标注任务

比如命名实体识别(NER)、中文分词(CWS)、词性标注(POS tagging) 等任务。

#### 3.1.1 CoNLL 格式

CoNLL 格式是一种在序列标注任务中广泛使用的格式。
通常情况下，此格式文本文件每行是一个词（或字）及其各种标签，连续的行组成一句话，多句话之间用 **空行** 隔开。
每行的第一列是词，最后一列是当前任务所使用的标签，一般是 `BIO` 和 `BIOES` 格式。

数据样例：

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

> 要使用 CoNLL 格式, 设置 `data_type: conll`。可选的，可以设置 `delimiter: ${custom_delimiter}`
> 来指定使用一个自定义的分隔符。默认使用空格或 `tab` （代码中传递了 `None` 给 `str.split`）。


#### 3.1.2 json-tags 格式

json-tags 格式与 CoNLL 类似，每个json对象需要包含 `'text'` 和 `'labels'` 两个键（也可以设置成其他名字，指定 `tags_key: ${tags_key}` 和 `text_key: ${text_key}`）。
'text' 和 'labels' 的长度必须一致，标签必须与词/字对应。

```
{
    "text": "鲁迅文学院组织有关专家",
    "labels": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O", "O"]
}
```

或者传入预先分好词的 `'text'`，例子如下。

```
{
    "text": ["鲁迅", "文学院", "组织", "有关", "专家"]
    "labels": ["B-ORG", "I-ORG", "O", "O", "O"]
}
```

> 若使用 CoNLL 格式，请设置 `data_type: json_tags`.


#### 3.1.3 json-spans 格式

json-spans 是另一种被 flat NER （普通的NER，实体词之间没有重叠） 和 nested NER （有重叠）。
每一个 span 由 `dict`表示，必须含有`start` `end` `type` 三个键，分别代表 span 的 [start, end) 位置（左闭右开），以及span的类别。

```
{
    "text": "鲁迅文学院组织有关专家",
    "spans": [{"start": 0, "end": 5, "type": "ORG"}, ...]
}
```

或者预先分好词的：

```
{
    "text": ["鲁迅", "文学院", "组织", "有关", "专家"]
    "spans": [{"start": 0, "end": 2, "type": "ORG"}, ...]
}
```

另外，我们支持 `type` 是 `List[str]`，即多个类别的列表，这常用于 entity typing 任务：

```
{
    "text": "人民日报出版社新近出版了王梦奎的短文集《翠微居杂笔》。",
    "spans": [{"start": 0, "end": 7, "type": ["组织", "出版商", "出版社"]}, ...]
}
```

> 要使用 json-spans 格式，请设置 `data_type: json_spans`.


#### 3.1.4 CLUENER 格式

CLUENER 格式是 CLUENER benchmark 的官方格式，把同一类型的实体聚合在了一起：

```
{
    "text": "鲁迅文学院组织有关专家",
    "label": {'ORG': [[0, 5], ...]}
}
```

> 若使用 CLUENER 格式，请设置 `data_type: cluener`.



# 4 数据集后处理

要让自定义的数据集（不是从内建 `DatasetBuilder` 加载的）可以直接用于模型训练，还需要一些后处理步骤：

1. dataset transform: 如果不是使用内建的 `DatasetBuilder` 脚本加载，还需要对数据集格式进行转换，让所含有的数据格式与相应任务的内建 `DatasetBuilder` 所返回的数据格式一致。
2. label counting: 使用任何数据集都需要统计相应的标签，目前内建支持了一些统计函数。

以上两种步骤均提供了一些必要的内建函数代码实现，如果需要扩展，也十分简单。
以上两个步骤均由 `adaseq/data/dataset_manager.py` 的 `DatasetManager` 执行。


## 4.1 数据集格式转换

> 目前仅支持NER数据集的格式转换，您可以自行扩展其他转换函数

Huggingface 和 modelscope 上常见的 NER 数据 都是前文所述的 `json-tags`, 分别有 `tokens` 和 `ner_tags` 两个键。

而 `adaseq` 目前的思想是将所有任务都统一成 span抽取，所以 `SequenceLabelingPreprocessor` 是从 spans 生成 `BIO/BIOES` 的标签序列。

所以，为了使用从 Huggingface 和 modelscope 加载的 NER 数据集，需要在数据集配置处指定 `transform` 参数，如下所示：

```yaml
dataset:
  path: conll2003
  transform:
    name: hf_ner_to_adaseq
    key: ner_tags
    scheme: bio
```

> `key` 指定了标签序列对应的字典键名，默认为 `'ner_tags'`。

> `scheme` 指定了标签序列的方案，目前仅支持 `BIO`。

**注意**: 目前仅提供了 `hf_ner_to_adaseq` 函数，其代码位于 `adaseq/data/utils.py`，您可以轻松的仿照此函数进行扩展，并将其设置到此代码文件中的全局变量 `DATASET_TRANSFORMS` 中，然后即可在配置文件中使用。

使用 `transform` 进行转换后，数据格式将与相应任务的内建 `DatasetBuilder` 所返回的数据格式一致。


## 4.2 标签集获取

为了给模型提供标签列表，`DatasetManager` 在初始化阶段会尝试获取数据的标签集，目前主要有3种方式：

1. 直接传入标签列表，设置 `labels: ['O', 'B-ORG', ...]`。
2. 传入标签文件路径或url，设置 `labels: PATH_OR_URL`，要求文件一行一个标签。
3. 从数据集统计，支持从数据样本字典的一个键值或样本的span中统计，具体见下文 4.2.1。

这几种方式都可在配置文件中指定，若没有设置，则默认从数据集统计，并根据不同的任务使用不同的统计方式。


### 4.2.1 从数据集统计标签

`DatasetManager` 会根据传入的统计函数设置，从数据的训练集和验证集 (train, dev/valid) 统计所有的标签，按字母排序 (`sorted`) 后生成 `label_to_id` 和 `id_to_label` 两个字典。
目前支持两种统计函数，见 `adaseq/data/utils.py` 的全局变量 `COUNT_LABEL_FUNCTIONS`。

第一个 `count_labels` 函数直接从数据样本字典的给定键 (`key` 参数，默认为 `'label'`) 统计标签，此键的值可以是 `str`（一个标签）或者是 `List[str]`，多个标签。
```yaml
dataset:
  ...
  labels:
    type: count_labels
    key: label
```

第而个 `count_span_labels` 函数直接从样本的每个span（span从给定 `key` 参数的键中获取，默认为 `'spans'`）的 `type` 中统计标签，可以是 `str`（一个标签）或者是 `List[str]`（多个标签）。
```yaml
dataset:
  ...
  labels:
    type: count_span_labels
    key: spans
```

您可以轻松的仿照以上函数进行扩展，并将其设置到全局变量 `COUNT_LABEL_FUNCTIONS` 中，然后即可在配置文件中使用。
