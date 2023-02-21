# FAQ

### 索引

<!-- TOC -->
 * [训练](#训练)
   * [1. 如何在自己的数据集上训练？](#t1)
   * [2. 有哪些预训练模型可以使用？](#t2)
   * [3. 本地训练，6G显存跑的起来么？](#t3)
   * [4. 如何在训练过程中保存除best之外的checkpoint？](#t4)
 * [推理&部署](#推理&部署)
   * [1. 有提供识别常用实体（人名、地名、组织）的基线模型吗？](#d1)
   * [2. 推理结果可以支持词典干预吗？](#d2)
   * [3. 如何转成onnx格式，便于实际生产环境中部署使用？](#d3)
<!-- TOC -->

### 训练

#### 1. 如何在自己的数据集上训练？
<a name="t1"></a>

> 参考[《自定义数据集》](https://github.com/modelscope/AdaSeq/blob/master/docs/tutorials/customizing_dataset_zh.md)，用户可以按照AdaSeq支持的格式整理数据，并通过本地文件的方式进行加载。如果无法转成相应的格式，也可以自定义数据加载脚本。

#### 2. 有哪些预训练模型可以使用？
<a name="t2"></a>

> AdaSeq的Embedder结构支持所有Encoder-Only的预训练模型，包括：
> - Huggingface上的backbone模型，如`bert-base-cased`, `xlm-roberta-large`, `sijunhe/nezha-cn-base`等
> - ModelScope上的backbone模型，如`damo/nlp_structbert_backbone_base_std`等
> - AdaSeq产出的任务模型，如`damo/nlp_raner_named-entity-recognition_chinese-base-news`等

#### 3. 本地训练，6G显存跑的起来么？
<a name="t3"></a>

> 根据模型方法、backbone、训练参数的不同组合占用的显存都不一样，如果GPU显存不足，建议可以尝试调低batch_size，并设置更小的max_length限制训练时的输入长度。

#### 4. 如何在训练过程中保存除best之外的checkpoint？
<a name="t4"></a>

> 可以在训练配置中指定使用CheckpointHook
> ```
> train:
>   hooks:
>     - type: CheckpointHook
>       interval: 1
> ```
> 其中interval为保存模型的频率，默认每1个epoch保存一次，更多细节详见[《回调函数机制详解》](https://modelscope.cn/docs/%E5%9B%9E%E8%B0%83%E5%87%BD%E6%95%B0%E6%9C%BA%E5%88%B6%E8%AF%A6%E8%A7%A3#CheckpointHook)。


### 推理&部署

#### 1. 有提供识别常用实体（人名、地名、组织）的基线模型吗？
<a name="d1"></a>

> 有的，[RaNER命名实体识别-中文-新闻领域-base](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-base-news/summary) 模型在ModelScope上可直接在线体验或者离线API调用。我们也提供了训练好的多语言、多行业、多规格的各种NER模型供使用：[查看所有NER模型](https://modelscope.cn/models?page=1&tasks=named-entity-recognition&type=nlp)。

#### 2. 推理结果可以支持词典干预吗？
<a name="d2"></a>

> AdaLA是我们开发的一款开箱即用的词法分析工具包，支持多粒度、多语言、多规格模型，支持高性能推理和自定义干预。目前正在内测中，敬请期待。

#### 3. 如何转成onnx格式，便于实际生产环境中部署使用？
<a name="d3"></a>

> 目前暂不支持转onnx。我们正在排期开发中。
