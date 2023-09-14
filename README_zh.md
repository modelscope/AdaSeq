# AdaSeq: An All-in-One Library for Developing State-of-the-Art Sequence Understanding Models

<div align="center">

[![license](https://img.shields.io/github/license/modelscope/adaseq.svg)](./LICENSE)
[![modelscope](https://img.shields.io/badge/modelscope->=1.4.0-624aff.svg)](https://modelscope.cn/)
![version](https://img.shields.io/github/tag/modelscope/adaseq.svg)
[![issues](https://img.shields.io/github/issues/modelscope/adaseq.svg)](https://github.com/modelscope/AdaSeq/issues)
[![stars](https://img.shields.io/github/stars/modelscope/adaseq.svg)](https://github.com/modelscope/AdaSeq/stargazers)
[![downloads](https://static.pepy.tech/personalized-badge/adaseq?period=total&left_color=grey&right_color=yellowgreen&left_text=downloads)](https://pypi.org/project/adaseq)
[![contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](./CONTRIBUTING.md)

</div>

<div align="center">

[English](./README.md) | 简体中文

</div>

## 简介
***AdaSeq*** (**A**libaba **D**amo **A**cademy **Seq**uence Understanding Toolkit) 是一个基于[ModelScope](https://modelscope.cn/home)的**一站式**序列理解开源工具箱，旨在提高开发者和研究者们的开发和创新效率，助力前沿论文工作落地。

![](./docs/imgs/task_examples_zh.png)

<details open>
<summary>🌟 <b>特性：</b></summary>

- **算法丰富**：

  AdaSeq提供了序列理解任务相关的大量前沿模型、训练方法和上下游工具。

- **性能强劲**：

  我们旨在开发性能最好的模型，在序列理解任务上胜出过其他开源框架。

- **简单易用**：

  只需一行命令，即可进行训练。

- **扩展性强**：

  用户可以自由注册模块组件，并通过组合不同的模块组件，便捷地构建自定义的检测模型。

</details>

⚠️**注意：** 这个项目仍在高速开发阶段，部分接口可能会发生改变。

## 📢 最新进展
- 2022-09: 欢迎尝试[SeqGPT](https://modelscope.cn/models/damo/nlp_seqgpt-560m/) - 零样本文本理解大模型
- 2022-07: [SemEval 2023] 我们U-RaNER论文获得了[最佳论文奖](https://semeval.github.io/SemEval2023/awards)!
- 2022-03: [SemEval 2023] 我们的U-RaNER模型赢得了[SemEval 2023多语言复杂实体识别比赛](https://multiconer.github.io/results) ***9个赛道的冠军***！[模型介绍和源代码](./examples/U-RaNER)！
- 2022-12: [[EMNLP 2022] 实现检索增强多模态实体理解MoRE模型](./examples/MoRe)
- 2022-11: [[EMNLP 2022] 实现超细粒度实体分类NPCRF模型](./examples/NPCRF)
- 2022-11: [[EMNLP 2022] 无监督边界感知预训练模型模型BABERT释出，实验复现](./examples/babert)

## ⚡ 快速体验
可以在ModelScope上快速体验我们的模型：
[[英文NER]](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_english-large-news/summary)
[[中文NER]](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-base-news/summary)
[[中文分词]](https://modelscope.cn/models/damo/nlp_structbert_word-segmentation_chinese-base/summary)

更多的任务、更多的语种、更多的领域：见全部已发布的模型卡片 [Modelcards](./docs/modelcards.md)

## 🛠️ 模型库
<details open>
<summary><b>支持的模型：</b></summary>

- [Transformer-based CRF](./examples/bert_crf)
- [Partial CRF](./examples/partial_bert_crf)
- [Retrieval Augmented NER](./examples/RaNER)
- [Biaffine NER](./examples/biaffine_ner)
- [Global-Pointer](./examples/global_pointer)
- [Multi-label Entity Typing](./examples/entity_typing)
- ...
</details>

## 💾 数据集
我们整理了很多序列理解相关任务的数据集：[Datasets](./docs/datasets.md)

## 📦 安装AdaSeq
AdaSeq项目基于 `Python version >= 3.7` 和 `PyTorch version >= 1.8`.

- pip安装：
```
pip install adaseq
```

- 源码安装：
```
git clone https://github.com/modelscope/adaseq.git
cd adaseq
pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

### 验证安装
为了验证AdaSeq是否安装成功，我们提供了一个demo配置文件用于训练模型（该文件需联网环境自动下载）。
```
adaseq train -c demo.yaml
```
运行过程中，你会看到不断刷新的训练日志；运行结束后，测试集评测结果将会被显示 `test: {"precision": xxx, "recall": xxx, "f1": xxx}`，同时在当前目录下，会生成`experiments/toy_msra/`，记录所有实验结果和保存的模型。

## 📖 教程文档
- [快速开始](./docs/tutorials/quick_start_zh.md)
- 基础教程
  - [了解配置文件](./docs/tutorials/learning_about_configs_zh.md)
  - [自定义数据集](./docs/tutorials/customizing_dataset_zh.md)
  - [TODO] 常用架构和模块化设计
- 最佳实践
  - [在自定义数据上训练模型](./docs/tutorials/training_a_model_zh.md)
  - [超参数优化](./docs/tutorials/hyperparameter_optimization_zh.md)
  - [训练加速](./docs/tutorials/training_acceleration_zh.md)
  - [模型推理](./docs/tutorials/model_inference_zh.md)
  - [模型发布到 ModelScope](./docs/tutorials/uploading_to_modelscope_zh.md)
  - [复现论文实验结果](./docs/tutorials/reproducing_papers_zh.md)
  - [TODO] 实现自定义模型
  - [TODO] 使用AdaLA进行推理
- [FAQ](./docs/faq_zh.md)

## 👫 开源社区
钉钉扫一扫，加入官方技术交流群。欢迎各位业界同好一起交流技术心得。

<div align="center">
<img src="./docs/imgs/community_qrcode.jpg" width="150"/>
<p>AdaSeq序列理解技术交流群</p>
</div>

## 📝 贡献指南
我们感谢所有为了改进AdaSeq而做的贡献，也欢迎社区用户积极参与到本项目中来。请参考 [CONTRIBUTING.md](./CONTRIBUTING.md) 来了解参与项目贡献的相关指引。

## 📄 开源许可证
本项目采用 Apache 2.0 License 开源许可证.
