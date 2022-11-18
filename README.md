# AdaSeq: An All-in-One Library for Developing State-of-the-Art Sequence Understanding Models
***AdaSeq*** (**A**libaba **D**amo **A**cademy **Seq**uence understanding toolkit) is an easy-to-use all-in-one library, built on [ModelScope](https://modelscope.cn/home), that allows researchers and developers to train custom models for sequence understanding tasks, including word segmentation, POS tagging, chunking, NER, entity typing, relation extraction, etc.

⚠️**Notice:** This project is under quick development. This means some interfaces could be changed in the future.

---

### Features:
- **State-of-the-Art**: we provide plenty of cutting-edge models, training methods and useful toolkits for sequence understanding tasks.
- **Easy-to-Use**: one line of command is all you need to obtain the best model.
- **Extensible**: easily register new tasks, models, modules, criterions, optimizers, lr_schedulers and training methods.

### What's New:
- 2022-11: [Released NPCRF code](./examples/NPCRF)
- 2022-11: [Released BABERT models](./examples/babert)

<details>
<summary>Previous updates</summary>
</details>

## State-of-the-Art Models
TODO

## Supported Models
- [Transformer-based CRF](./examples/bert_crf)
- [Partial CRF](./examples/partial_bert_crf)
- [Retrieval Augmented NER](./examples/RaNER)
- [Global-Pointer](./examples/global_pointer)
- [Multi-label Entity Typing](./examples/entity_typing)
- ...

## Quick Start
### Requirements and Installation
Python version >= 3.7
```
git clone https://github.com/modelscope/adaseq.git
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
- Tutorial 1: [Training a Model & Configuration Explanation](./docs/tutorials/training_a_model.md)
- Tutorial 2: [Preparing Custom Dataset](./docs/tutorials/preparing_custom_dataset.md)
- Tutorial 3: [Hyperparameter Tuning with Grid Search](./docs/tutorials/hyperparameter_tuning_with_grid_search.md)
- Tutorial 4: [Training with Multiple GPUs](./docs/tutorials/training_with_multiple_gpus.md)

## License
This project is licensed under the Apache License (Version 2.0).
