# Bert-CRF
The Transformer-based CRF (or simply Bert-CRF) model is widely used in sequence labeling tasks and has proven to be a strong baseline in most scenarios. Proudly, we can say that, due to the careful model design and training techniques, our implementation can beat most off-the-shelf sequence labeling frameworks.

This example introduces how to train a Transformer-based CRF model using AdaSeq.

---

## Training a new Model
Let's train a NER model using the preset script as an example.
```
python scripts/train.py -c examples/bert_crf/configs/resume.yaml
```

<details>
<summary>See details for training</summary>

### Model-specific Arguments
```
preprocessor:
  tag_scheme: BIOES  # (str, optional): The tag scheme used for sequence-labeling tasks. Possible candidates are [`BIO`, `BIOES`]. Default to `BIOES`.
model:
  word_dropout: 0.1  # (float, optional): Word-level/token-level dropout probability. Default to `0`.
  use_crf: true  # (bool, optional): Whether to use CRF decoder. Default to `true`.
```

### Using Custom Dataset
See [tutorial](../../docs/tutorials/customizing_dataset.md)
</details>

## Benchmarks

| Language |    Dataset    |     Backbone      | AdaSeq Bert-CRF |                                Best published                                 |                                                   Modelcard & Demo                                                    |
|:--------:|:-------------:|:-----------------:|:---------------:|:-----------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| Chinese  |     msra      |  structbert-base  |      96.69      |   96.72 [(Li et al., 2020)](https://aclanthology.org/2020.acl-main.45.pdf)    |     [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-base-news/summary)      |
| Chinese  | ontonotes 4.0 |  structbert-base  |      83.04      |   84.47 [(Li et al., 2020)](https://aclanthology.org/2020.acl-main.45.pdf)    |    [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-base-generic/summary)    |
| Chinese  |    resume     |  structbert-base  |      96.87      |      96.79 [(Xuan et al., 2020)](https://arxiv.org/pdf/2001.05272v6.pdf)      |    [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-base-resume/summary)     |
| Chinese  |     weibo     |  structbert-base  |      72.77      |  72.66 [(Zhu et al., 2022)](https://aclanthology.org/2022.acl-long.490.pdf)   | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-base-social_media/summary)  |
| English  |    conll03    | xlm-roberta-large |      93.35      |           94.6 [(Wang pap, 2021)](https://arxiv.org/abs/2010.05006)           |     [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_english-large-news/summary)     |
| English  |    conllpp    | xlm-roberta-large |      94.71      | 95.88 [(Zhou et al., 2021)](https://aclanthology.org/2021.emnlp-main.437.pdf) |     [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_english-large-news/summary)     |
| English  |    wnut16     | xlm-roberta-large |      57.23      |         58.98 [(Wang et al., 2021)](https://arxiv.org/abs/2105.03654)         |                                                           -                                                           |
| English  |    wnut17     | xlm-roberta-large |      59.69      |         60.45 [(Wang et al., 2021)](https://arxiv.org/abs/2105.03654)         | [ModelScope](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_english-large-social_media/summary) |
