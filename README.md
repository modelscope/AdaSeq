# AdaSeq: An All-in-One Library for Developing State-of-the-Art Sequence Understanding Models
***AdaSeq*** (**A**libaba **D**amo **A**cademy **Seq**uence understanding toolkit) is an easy-to-use all-in-one toolkit that allows researchers and developers to train custom models for sequence understanding tasks, including word segmentation, POS tagging, chunking, NER, entity typing, relation extraction, etc.

---

### Features

### What's New

## State-of-the-Art Models

## Quick Start
### Requirements
```
pip install -r requirements.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

### train a model
```
python scripts/train.py -c examples/bert_crf/configs/resume.yaml
```

### test a model
```
python scripts/test.py -c examples/bert_crf/configs/resume.yaml -cp ${checkpoint_path}
```

## Tutorials
- Tutorial 1: Train a model & How to writing a config file
- Tutorial 2: Hyperparameter Tuning with Grid Search
- Tutorial 3: Train with multiple gpus

## Contributing
1. [UNER开发规约](https://yuque.antfin-inc.com/docs/share/7088e485-5817-4beb-8a28-f8de7dd95a9a?# 《UNER开发规约》)
2. [Modelscope文档中心](https://modelscope.cn/docs/%E9%A6%96%E9%A1%B5)

## License
This project is licensed under the Apache License (Version 2.0).
